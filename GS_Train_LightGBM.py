import os
import pandas as pd
import pickle
import numpy as np
from lightgbm import LGBMRegressor
from GS_Functions import calculate_rrmse, calculate_rmse, calculate_mape
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
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
                    # 'time' 열을 datetime 형식으로 변환
                    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

                    # Trip 구분을 위해 각 데이터에 파일 인덱스(Trip ID) 추가
                    data['trip_id'] = files.index(file)

                    data['mean_accel_10'] = data['acceleration'].rolling(window=5).mean().bfill()
                    data['std_accel_10'] = data['acceleration'].rolling(window=5).std().bfill()
                    data['mean_speed_10'] = data['speed'].rolling(window=5).mean().bfill()
                    data['std_speed_10'] = data['speed'].rolling(window=5).std().bfill()

                    df_list.append(data)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    full_data = pd.concat(df_list, ignore_index=True)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 스케일링 범위에 'elapsed_time'의 최소값 0초, 최대값 21600초로 추가
        scaler.fit(pd.DataFrame([
            [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN, 0, 0, 0, 0],
            [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX, 1, 1, 1, 1]
        ], columns=['speed', 'acceleration', 'ext_temp', 'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10']))

    # 모든 피쳐에 대해 스케일링 적용
    full_data[['speed', 'acceleration', 'ext_temp', 'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10']] = scaler.transform(
        full_data[['speed', 'acceleration', 'ext_temp', 'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10']])

    return full_data, scaler

def integrate_and_compare(trip_data):
    # 'time'을 기준으로 정렬
    trip_data = trip_data.sort_values(by='time')

    # 'time'을 초 단위로 변환
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # 'Power_phys + y_pred' 적분 (trapz 사용)
    trip_data['Power_hybrid'] = trip_data['Power_phys'] + trip_data['y_pred']
    hybrid_integral = np.trapz(trip_data['Power_hybrid'].values, time_seconds)

    # 'Power_data' 적분 (trapz 사용)
    data_integral = np.trapz(trip_data['Power_data'].values, time_seconds)

    # 적분된 값 반환
    return hybrid_integral, data_integral

def grid_search_lambda(X_train, y_train):
    # Define a range of lambda values for grid search
    lambda_values = np.logspace(-3, 7, num=11)
    param_grid = {
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'lambda_l2': lambda_values  # L2 regularization term in LightGBM
    }

    model = LGBMRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    best_lambda = grid_search.best_params_['lambda_l2']
    print(f"Best lambda found: {best_lambda}")

    return best_lambda

def cross_validate(vehicle_files, selected_car, precomputed_lambda, plot=None, save_dir="models"):
    model_name = "LGBM"  # Update the model name for LightGBM

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

        # Process train and test data
        train_data, scaler = process_files(train_files)
        test_data, _ = process_files(test_files)

        # Prepare training data
        X_train = train_data[['speed', 'acceleration', 'ext_temp', 'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10']].to_numpy()
        y_train = train_data['Residual'].to_numpy()

        X_test = test_data[['speed', 'acceleration', 'ext_temp', 'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10']].to_numpy()
        y_test = test_data['Residual'].to_numpy()

        # Find best lambda if not precomputed
        if best_lambda is None:
            best_lambda = grid_search_lambda(X_train, y_train)

        # Train LGBM model with best lambda
        model = LGBMRegressor(lambda_l2=best_lambda, learning_rate=0.1, n_estimators=200)
        model.fit(X_train, y_train)

        # Predict for both train and test data
        train_data['y_pred'] = model.predict(X_train)
        test_data['y_pred'] = model.predict(X_test)

        # Evaluate model by integrating and comparing results
        train_trip_groups = train_data.groupby('trip_id')
        test_trip_groups = test_data.groupby('trip_id')

        hybrid_integrals_train, data_integrals_train = [], []
        for _, group in train_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_train.append(hybrid_integral)
            data_integrals_train.append(data_integral)

        mape_train = calculate_mape(np.array(data_integrals_train), np.array(hybrid_integrals_train))
        rrmse_train = calculate_rrmse(np.array(data_integrals_train), np.array(hybrid_integrals_train))

        hybrid_integrals_test, data_integrals_test = [], []
        for _, group in test_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_test.append(hybrid_integral)
            data_integrals_test.append(data_integral)

        mape_test = calculate_mape(np.array(data_integrals_test), np.array(hybrid_integrals_test))
        rrmse_test = calculate_rrmse(np.array(data_integrals_test), np.array(hybrid_integrals_test))

        # RMSE Calculation
        rmse = calculate_rmse((y_test + test_data['Power_phys']), (test_data['y_pred'] + test_data['Power_phys']))

        results.append((fold_num, rmse, rrmse_train, mape_train, rrmse_test, mape_test))
        models.append(model)

        print(f"Vehicle: {selected_car}, Fold: {fold_num}")
        print(f"Train - MAPE: {mape_train:.2f}, RRMSE: {rrmse_train:.2f}")
        print(f"Test - MAPE: {mape_test:.2f}, RRMSE: {rrmse_test:.2f}")

    # Select the best model based on test set MAPE
    median_mape = np.median([result[5] for result in results])
    median_index = np.argmin([abs(result[3] - median_mape) for result in results])
    best_model = models[median_index]

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save the best model
        if best_model:
            model_file = os.path.join(save_dir, f"{model_name}_best_model_{selected_car}.txt")
            best_model.booster_.save_model(model_file)
            print(f"Best model for {selected_car} saved with MAPE: {median_mape}")

        # Save the scaler
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler_{selected_car}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at {scaler_path}")

    return results, scaler, best_lambda
