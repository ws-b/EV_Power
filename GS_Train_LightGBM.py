import os
import pandas as pd
import pickle
import numpy as np
import lightgbm as lgb
from GS_Functions import calculate_rrmse, calculate_rmse, calculate_mape
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from GS_plot import plot_3d, plot_contour
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_file(file):
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            return data[['time', 'speed', 'acceleration', 'ext_temp', 'Residual', 'Power_phys', 'Power_data']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None

def process_files(files, scaler=None):
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6 # 230km/h 를 m/s 로
    ACCELERATION_MIN = -15 # m/s^2
    ACCELERATION_MAX = 9 # m/s^2
    TEMP_MIN = -30
    TEMP_MAX = 50

    df_list = []
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                data = future.result()
                if data is not None:
                    # Calculate absolute acceleration
                    data['abs_acceleration'] = data['acceleration'].abs()

                    # Calculate rolling mean and standard deviation for the last 5 rows
                    data['mean_abs_accel_5'] = data['abs_acceleration'].rolling(window=5).mean()
                    data['std_abs_accel_5'] = data['abs_acceleration'].rolling(window=5).std()
                    data['mean_speed_5'] = data['speed'].rolling(window=5).mean()
                    data['std_speed_5'] = data['speed'].rolling(window=5).std()

                    # Forward fill to replace NaNs with the first available value
                    data[['mean_abs_accel_5', 'std_abs_accel_5', 'mean_speed_5', 'std_speed_5']] = data[['mean_abs_accel_5', 'std_abs_accel_5', 'mean_speed_5', 'std_speed_5']].ffill()

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
        scaler.fit(pd.DataFrame([[SPEED_MIN, ACCELERATION_MIN, TEMP_MIN], [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX]], columns=['speed', 'acceleration','ext_temp']))

    # Scale the features
    full_data[['speed', 'acceleration', 'ext_temp']] = scaler.transform(full_data[['speed', 'acceleration', 'ext_temp']])

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

def grid_search_lambda(X_train, y_train):
    dtrain = lgb.Dataset(X_train, label=y_train, weight=X_train[:, 0])
    lambda_values = np.logspace(-3, 7, num=11)
    param_grid = {
        'boosting_type': ['gbdt'],
        'objective': ['regression'],
        'metric': ['rmse'],
        'lambda_l2': lambda_values
    }

    model = lgb.LGBMRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    best_lambda = grid_search.best_params_['lambda_l2']
    print(f"Best lambda found: {best_lambda}")

    return best_lambda

def cross_validate(vehicle_files, selected_car, precomputed_lambda, plot=None, save_dir="models"):
    model_name = "LightGBM"

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []
    best_model = None

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]

    best_lambda = precomputed_lambda
    for fold_num, (train_index, test_index) in enumerate(kf.split(files), 1):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        train_data, scaler = process_files(train_files)
        test_data, _ = process_files(test_files)

        # Include the new features in the training and test sets
        X_train = train_data[['speed', 'acceleration', 'ext_temp', 'mean_abs_accel_5', 'std_abs_accel_5', 'mean_speed_5', 'std_speed_5']].to_numpy()
        y_train = train_data['Residual'].to_numpy()

        X_test = test_data[['speed', 'acceleration', 'ext_temp', 'mean_abs_accel_5', 'std_abs_accel_5', 'mean_speed_5', 'std_speed_5']].to_numpy()
        y_test = test_data['Residual'].to_numpy()

        if best_lambda is None:
            best_lambda = grid_search_lambda(X_train, y_train)

        dtrain = lgb.Dataset(X_train, label=y_train, weight=X_train[:, 0])
        dtest = lgb.Dataset(X_test, label=y_test, weight=X_test[:, 0])

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'lambda_l2': best_lambda
        }

        evals_result = {}
        model = lgb.train(params, dtrain, num_boost_round=150, valid_sets=[dtrain, dtest], evals_result=evals_result)
        y_pred = model.predict(X_test)

        mape = calculate_mape(y_test + test_data['Power_phys'], y_pred + test_data['Power_phys'])
        rmse = calculate_rmse((y_test + test_data['Power_phys']), (y_pred + test_data['Power_phys']))
        rrmse = calculate_rrmse((y_test + test_data['Power_phys']), (y_pred + test_data['Power_phys']))
        residual2 = y_test - y_pred
        results.append((fold_num, rrmse, rmse, mape))
        models.append(model)
        print(f"Vehicle: {selected_car}, Fold: {fold_num}, RMSE: {rmse}, RRMSE: {rrmse}, MAPE: {mape}")

        # Calculate the median RRMSE
        median_rrmse = np.median([result[1] for result in results])
        # Find the index of the model corresponding to the median RRMSE
        median_index = np.argmin([abs(result[1] - median_rrmse) for result in results])
        best_model = models[median_index]

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save the best model
        if best_model:
            model_file = os.path.join(save_dir, f"{model_name}_best_model_{selected_car}.txt")
            best_model.save_model(model_file)
            print(f"Best model for {selected_car} saved with RRMSE: {median_rrmse}")

        # Save the scaler
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler_{selected_car}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at {scaler_path}")

    return results, scaler, best_lambda



def process_file_with_trained_model(file, model, scaler):
    try:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns and 'Power_phys' in data.columns:
            # Calculate absolute acceleration
            data['abs_acceleration'] = data['acceleration'].abs()

            # Calculate rolling mean and standard deviation for the last 5 rows
            data['mean_abs_accel_5'] = data['abs_acceleration'].rolling(window=5).mean()
            data['std_abs_accel_5'] = data['abs_acceleration'].rolling(window=5).std()
            data['mean_speed_5'] = data['speed'].rolling(window=5).mean()
            data['std_speed_5'] = data['speed'].rolling(window=5).std()

            # Forward fill to replace NaNs with the first available value
            data[['mean_abs_accel_5', 'std_abs_accel_5', 'mean_speed_5', 'std_speed_5']] = data[['mean_abs_accel_5', 'std_abs_accel_5', 'mean_speed_5', 'std_speed_5']].ffill()

            # Use the provided scaler to scale all necessary features
            features = data[['speed', 'acceleration', 'ext_temp', 'mean_abs_accel_5', 'std_abs_accel_5', 'mean_speed_5', 'std_speed_5']]
            features_scaled = scaler.transform(features)

            # Predict the residual using the trained model
            predicted_residual = model.predict(features_scaled)

            # Calculate the hybrid power
            data['Power_hybrid'] = predicted_residual + data['Power_phys']

            # Save the updated file
            data.to_csv(file, index=False)

            print(f"Processed file {file}")
        else:
            print(f"File {file} does not contain required columns 'speed', 'acceleration', or 'Power_phys'.")
    except Exception as e:
        print(f"Error processing file {file}: {e}")

def add_predicted_power_column(files, model_path, scaler):
    try:
        model = lgb.Booster(model_file=model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_with_trained_model, file, model, scaler) for file in files]
        for future in as_completed(futures):
            future.result()