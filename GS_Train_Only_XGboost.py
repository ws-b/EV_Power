import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from GS_plot import plot_3d, plot_contour
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_file(file):
    try:
        data = pd.read_csv(file)
        if 'Power' in data.columns and 'Power_IV' in data.columns:
            return data[['speed', 'acceleration', 'Power_IV']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None

def process_files(files, scaler=None):
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

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame([[SPEED_MIN, ACCELERATION_MIN], [SPEED_MAX, ACCELERATION_MAX]], columns=['speed', 'acceleration']))

    full_data[['speed', 'acceleration']] = scaler.transform(full_data[['speed', 'acceleration']])

    return full_data, scaler

def calculate_rrmse(y_test, y_pred):
    relative_errors = (y_pred - y_test) / y_test

    rrmse = np.sqrt(np.mean(relative_errors ** 2))

    return rrmse

def grid_search_lambda(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=X_train[:, 0])
    lambda_values = np.logspace(-3, 7, num=11)
    param_grid = {
        'tree_method': ['hist'],
        'device': ['cuda'],
        'eval_metric': ['rmse'],
        'lambda': lambda_values
    }

    model = xgb.XGBRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
    grid_search.fit(X_train, y_train)
    best_lambda = grid_search.best_params_['lambda']

    print(f"Best lambda found: {best_lambda}")
    return best_lambda

def cross_validate(vehicle_files, selected_car, save_dir="models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_name = "XGB_Only"

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []
    best_model = None

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]
    # 전체 데이터를 사용하여 평균 계산
    full_data, scaler = process_files(files)
    y = full_data['Power_IV'].to_numpy()

    for fold_num, (train_index, test_index) in enumerate(kf.split(files), 1):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        train_data, scaler = process_files(train_files)
        test_data, _ = process_files(test_files)

        X_train = train_data[['speed', 'acceleration']].to_numpy()
        y_train = train_data['Power_IV'].to_numpy()

        X_test = test_data[['speed', 'acceleration']].to_numpy()
        y_test = test_data['Power_IV'].to_numpy()

        best_lambda = grid_search_lambda(X_train, y_train)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=X_train[:, 0])
        dtest = xgb.DMatrix(X_test, label=y_test, weight=X_test[:, 0])

        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': ['rmse'],
            'lambda': best_lambda
        }

        evals = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(params, dtrain, num_boost_round=150, evals=evals)
        y_pred = model.predict(dtest)
        rrmse = calculate_rrmse(y_pred, y_test)
        results.append((fold_num, rrmse))
        models.append(model)
        print(f"Vehicle: {selected_car}, Fold: {fold_num}, RRMSE: {rrmse}")

        # Calculate the median RRMSE
        median_rrmse = np.median([result[1] for result in results])
        # Find the index of the model corresponding to the median RRMSE
        median_index = np.argmin([abs(result[1] - median_rrmse) for result in results])
        best_model = models[median_index]

    # Save the best model
    if best_model:
        model_file = os.path.join(save_dir, f"{model_name}_best_model_{selected_car}.json")
        surface_plot = os.path.join(save_dir, f"{model_name}_best_model_{selected_car}_plot.html")
        best_model.save_model(model_file)
        print(f"Best model for {selected_car} saved with RRMSE: {median_rrmse}")

        Residual = y_pred - y_test
        plot_contour(X_test, Residual, scaler, selected_car, '(Predicted Power - BMS Power)', num_grids=400)

    # Save the scaler
    scaler_path = os.path.join(save_dir, f'{model_name}_scaler_{selected_car}.pkl')
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

            data['Predicted_Power'] = predicted_residual

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