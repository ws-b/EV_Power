import os
import pandas as pd
import numpy as np
import pickle
import joblib
from GS_Functions import calculate_rrmse, calculate_rmse, calculate_mape
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from GS_plot import plot_contour
from concurrent.futures import ProcessPoolExecutor, as_completed

# 데이터 전처리 함수
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
    feature_cols = ['speed', 'acceleration', 'ext_temp', 'mean_accel_10', 'std_accel_10', 'mean_speed_10',
                    'std_speed_10']
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
        ], columns=feature_cols))

    # 모든 피쳐에 대해 스케일링 적용
    full_data[feature_cols] = scaler.transform(
        full_data[feature_cols])

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
# 교차 검증 및 모델 학습 함수
def cross_validate(vehicle_files, selected_car, plot=None, save_dir="models"):
    model_name = "LR"
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []
    best_model = None

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]

    for fold_num, (train_index, test_index) in enumerate(kf.split(files), 1):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        # Train set과 test set을 처리
        train_data, scaler = process_files(train_files)
        test_data, _ = process_files(test_files)

        # 각 Trip별로 그룹화하여 적분 수행
        train_trip_groups = train_data.groupby('trip_id')
        test_trip_groups = test_data.groupby('trip_id')

        feature_cols = ['speed', 'acceleration', 'ext_temp', 'mean_accel_10', 'std_accel_10', 'mean_speed_10',
                        'std_speed_10']
        # 학습에 사용할 데이터 준비
        X_train = train_data[feature_cols].to_numpy()
        y_train = train_data['Residual'].to_numpy()

        X_test = test_data[feature_cols].to_numpy()
        y_test = test_data['Residual'].to_numpy()

        # Linear Regression 모델 학습
        model = LinearRegression()
        model.fit(X_train, y_train)
        train_data['y_pred'] = model.predict(X_train)
        test_data['y_pred'] = model.predict(X_test)

        # Train set에서 Trip별로 적분을 계산하고 MAPE 및 RRMSE 계산
        hybrid_integrals_train, data_integrals_train = [], []
        for _, group in train_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_train.append(hybrid_integral)
            data_integrals_train.append(data_integral)

        # MAPE 및 RRMSE 계산 (Train set)
        mape_train = calculate_mape(np.array(data_integrals_train), np.array(hybrid_integrals_train))
        rrmse_train = calculate_rrmse(np.array(data_integrals_train), np.array(hybrid_integrals_train))

        # Test set에서도 동일한 방식으로 MAPE 및 RRMSE 계산
        hybrid_integrals_test, data_integrals_test = [], []
        for _, group in test_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_test.append(hybrid_integral)
            data_integrals_test.append(data_integral)

        # MAPE 및 RRMSE 계산 (Test set)
        mape_test = calculate_mape(np.array(data_integrals_test), np.array(hybrid_integrals_test))
        rrmse_test = calculate_rrmse(np.array(data_integrals_test), np.array(hybrid_integrals_test))

        # RMSE 계산 (Test set)
        rmse = calculate_rmse((y_test + test_data['Power_phys']), (test_data['y_pred'] + test_data['Power_phys']))

        # 결과 저장
        results.append((fold_num, rmse, rrmse_train, mape_train, rrmse_test, mape_test))
        models.append(model)

        # 각 Fold별 결과 출력
        print(f"Vehicle: {selected_car}, Fold: {fold_num}")
        print(f"Train - MAPE: {mape_train:.2f}, RRMSE: {rrmse_train:.2f}")
        print(f"Test  - MAPE: {mape_test:.2f}, RRMSE: {rrmse_test:.2f}")

    # 최종 모델 선택 (Test set에서 RRMSE 기준으로 중간값에 해당하는 모델)
    median_mape = np.median([result[5] for result in results])
    median_index = np.argmin([abs(result[4] - median_mape) for result in results])
    best_model = models[median_index]

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 최적 모델 저장
        if best_model:
            model_file = os.path.join(save_dir, f"{model_name}_best_model_{selected_car}.joblib")
            joblib.dump(best_model, model_file)
            print(f"Best model for {selected_car} saved with MAPE: {median_mape}")

        # 스케일러 저장
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler_{selected_car}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at {scaler_path}")

    return results, scaler