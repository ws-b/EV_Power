import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from GS_plot import plot_3d, plot_contour
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_file(file):
    try:
        data = pd.read_csv(file)
        if 'Power' in data.columns and 'Power_IV' in data.columns:
            data['Residual'] = data['Power'] - data['Power_IV']
            return data[['speed', 'acceleration', 'Residual']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None

def process_files(files):
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6 # 230km/h 를 m/s 로
    ACCELERATION_MIN = -15 # m/s^2
    ACCELERATION_MAX = 9 # m/s^2

    df_list = []
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                data = future.result()
                if data is not None:
                    df_list.append((files.index(file), data))
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    # Sort the list by the original file order
    df_list.sort(key=lambda x: x[0])
    df_list = [df for _, df in df_list]

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    full_data = pd.concat(df_list, ignore_index=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(pd.DataFrame([[SPEED_MIN, ACCELERATION_MIN], [SPEED_MAX, ACCELERATION_MAX]], columns=['speed', 'acceleration']))

    full_data[['speed', 'acceleration']] = scaler.transform(full_data[['speed', 'acceleration']])

    return full_data, scaler


def custom_obj(preds, dtrain):
    labels = dtrain.get_label()
    speed = dtrain.get_weight()  # Use weight to store speed

    grad = preds - labels
    hess = np.ones_like(grad)

    # speed가 0인 경우 제약 조건 반영
    mask = (speed == 0)
    grad[mask] = np.maximum(0, grad[mask])

    return grad, hess

def cross_validate(vehicle_files, selected_car, save_dir="models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_rmse = float('inf')
    best_model = None

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]
    data, scaler = process_files(files)
    X = data[['speed', 'acceleration']].to_numpy()
    y = data['Residual'].to_numpy()

    y_mean = np.mean(y)

    for fold_num, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=X_train[:, 0])
        dtest = xgb.DMatrix(X_test, label=y_test, weight=X_test[:, 0])

        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': ['rmse'],
            'lambda' : 1
        }
        evals = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(params, dtrain, num_boost_round=150, evals=evals, obj=custom_obj)
        y_pred = model.predict(dtest)
        residual2 = y_test - y_pred
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nrmse = rmse / y_mean
        results.append((fold_num, rmse, nrmse))
        print(f"Vehicle: {selected_car}, Fold: {fold_num}, RMSE: {rmse}, NRMSE: {nrmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    # Save the best model
    if best_model:
        model_file = os.path.join(save_dir, f"XGB_best_model_{selected_car}.json")
        surface_plot = os.path.join(save_dir, f"XGB_best_model_{selected_car}_plot.html")
        best_model.save_model(model_file)
        print(f"Best model for {selected_car} saved with RMSE: {best_rmse}")
        plot_3d(X_test, y_test, y_pred, fold_num, selected_car, scaler, 400, 30, output_file=surface_plot)

        plot_contour(X_test, y_pred, scaler, selected_car, 'Predicted Residual[1]', num_grids=400)
        plot_contour(X_test, residual2, scaler, selected_car, 'Residual[2]',  num_grids=400)

    # Save the scaler
    scaler_path = os.path.join(save_dir, f'XGB_scaler_{selected_car}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved at {scaler_path}")

    return results, scaler


def process_file_with_trained_model(file, model, scaler):
    try:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns and 'Power' in data.columns:
            # Use the provided scaler
            features = data[['speed', 'acceleration']]
            features_scaled = scaler.transform(features)

            predicted_residual = model.predict(features_scaled)

            data['Predicted_Power'] = data['Power'] - predicted_residual

            # Save the updated file
            data.to_csv(file, index=False)

            print(f"Processed file {file}")
        else:
            print(f"File {file} does not contain required columns 'speed', 'acceleration', or 'Power'.")
    except Exception as e:
        print(f"Error processing file {file}: {e}")

def add_predicted_power_column(files, model_path, scaler):
    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_with_trained_model, file, model, scaler) for file in files]
        for future in as_completed(futures):
            future.result()
