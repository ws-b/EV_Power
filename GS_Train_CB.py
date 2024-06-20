import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
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
    SPEED_MAX = 230 / 3.6
    ACCELERATION_MIN = -15
    ACCELERATION_MAX = 9

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
    scaler.fit(
        pd.DataFrame([[SPEED_MIN, ACCELERATION_MIN], [SPEED_MAX, ACCELERATION_MAX]], columns=['speed', 'acceleration']))

    full_data[['speed', 'acceleration']] = scaler.transform(full_data[['speed', 'acceleration']])

    return full_data, scaler


def cross_validate(vehicle_files, selected_car, save_dir="models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_rmse = float('inf')
    best_model = None

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
    return None, None

    files = vehicle_files[selected_car]
    data, scaler = process_files(files)
    X = data[['speed', 'acceleration']].to_numpy()
    y = data['Residual'].to_numpy()

    y_range = np.ptp(y)

    for fold_num, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_pool = Pool(X_train, y_train)
        test_pool = Pool(X_test, y_test)

        model = CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            loss_function='RMSE',
            task_type='GPU',
            eval_metric='RMSE',
            verbose=0
        )

        model.fit(train_pool, eval_set=test_pool)

        y_pred = model.predict(X_test)

        # 속도가 0인 경우 y_pred가 음수인지 확인하고, 음수인 경우 0으로 설정
        speed_zero_mask = (X_test[:, 0] == 0)
        y_pred[speed_zero_mask] = np.maximum(y_pred[speed_zero_mask], 0)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nrmse = rmse / y_range
        results.append((fold_num, rmse, nrmse))
        print(f"Vehicle: {selected_car}, Fold: {fold_num}, RMSE: {rmse}, NRMSE: {nrmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    # Save the best model
    if best_model:
        model_file = os.path.join(save_dir, f"CatBoost_best_model_{selected_car}.cbm")
        surface_plot = os.path.join(save_dir, f"CatBoost_best_model_{selected_car}_plot.html")
        best_model.save_model(model_file)
        print(f"Best model for {selected_car} saved with RMSE: {best_rmse}")
        plot_3d(X_test, y_test, y_pred, fold_num, selected_car, scaler, 400, 30, output_file=surface_plot)
        plot_contour(X_test, y_pred, scaler, selected_car, num_grids=400, output_file=None)

    return results, scaler


def process_file_with_trained_model(file, model, scaler):
    try:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns and 'Power' in data.columns:
            # Use the provided scaler
            features = data[['speed', 'acceleration']]
            features_scaled = scaler.transform(features)

            predicted_residual = model.predict(features_scaled)

            # 속도가 0인 경우 y_pred가 음수인지 확인하고, 음수인 경우 0으로 설정
            speed_zero_mask = (features['speed'] == 0)
            predicted_residual[speed_zero_mask] = np.maximum(predicted_residual[speed_zero_mask], 0)

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
        model = CatBoostRegressor()
        model.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_with_trained_model, file, model, scaler) for file in files]
        for future in as_completed(futures):
            future.result()
