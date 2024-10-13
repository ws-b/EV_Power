import os
import pandas as pd
import pickle
import numpy as np
# import optuna  # No longer needed
import joblib
import time  # Import time module for tracking execution time
from GS_Functions import calculate_rrmse, calculate_rmse, calculate_mape
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from GS_plot import plot_contour, plot_composite_contour  # plot_shap_values 함수는 생략합니다.
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


# ----------------------------
# 데이터 처리 함수
# ----------------------------
def process_single_file(file):
    """
    단일 CSV 파일을 처리하여 잔차를 계산하고 관련 열을 선택합니다.
    """
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            return data[['time', 'speed', 'acceleration', 'ext_temp', 'Residual', 'Power_phys', 'Power_data']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None


def process_files(files, scaler=None):
    """
    여러 CSV 파일을 병렬로 처리하고, 롤링 통계량을 계산하며 특징을 스케일링합니다.
    """
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6  # km/h를 m/s로 변환
    ACCELERATION_MIN = -15  # m/s^2
    ACCELERATION_MAX = 9  # m/s^2
    TEMP_MIN = -30
    TEMP_MAX = 50
    feature_cols = [
        'speed', 'acceleration', 'ext_temp',
        'mean_accel_10', 'std_accel_10',
        'mean_speed_10', 'std_speed_10'
    ]

    df_list = []
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                data = future.result()
                if data is not None:
                    # 'time' 열을 datetime으로 변환
                    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

                    # 트립 구분을 위한 trip_id 추가
                    data['trip_id'] = files.index(file)

                    # 윈도우 크기 5로 롤링 통계량 계산
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
        scaler.fit(pd.DataFrame([
            [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN, 0, 0, 0, 0],
            [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX, 1, 1, 1, 1]
        ], columns=feature_cols))

    # 모든 특징에 스케일링 적용
    full_data[feature_cols] = scaler.transform(full_data[feature_cols])

    return full_data, scaler


def integrate_and_compare(trip_data):
    """
    트립 데이터에서 'Power_hybrid'와 'Power_data'를 시간에 따라 적분합니다.
    """
    # 'time'으로 정렬
    trip_data = trip_data.sort_values(by='time')

    # 'time'을 초 단위로 변환
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # 'Power_phys + y_pred'를 트래피조이드 룰로 적분
    trip_data['Power_hybrid'] = trip_data['Power_phys'] + trip_data['y_pred']
    hybrid_integral = np.trapz(trip_data['Power_hybrid'].values, time_seconds)

    # 'Power_data'를 트래피조이드 룰로 적분
    data_integral = np.trapz(trip_data['Power_data'].values, time_seconds)

    return hybrid_integral, data_integral


# ----------------------------
# 모델 훈련 함수
# ----------------------------

def train_model(X_train, y_train, best_params):
    """
    최적의 하이퍼파라미터로 SVR 모델을 훈련시킵니다.
    """
    # SVR 모델 생성
    model = SVR(
        C=best_params['C'],
        epsilon=best_params['epsilon'],
        gamma=best_params['gamma'],
        kernel=best_params['kernel']
    )

    # 모델 훈련
    model.fit(X_train, y_train)

    return model


# ----------------------------
# 교차 검증 및 모델 훈련
# ----------------------------

def cross_validate(vehicle_files, selected_car, params=None, plot=False):
    model_name = "SVR"

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []
    best_params_overall = None

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car][:499]

    for fold_num, (train_index, test_index) in enumerate(kf.split(files), 1):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        # 훈련 및 테스트 데이터 처리
        train_data, scaler = process_files(train_files)
        test_data, _ = process_files(test_files, scaler=scaler)

        feature_cols = [
            'speed', 'acceleration', 'ext_temp',
            'mean_accel_10', 'std_accel_10',
            'mean_speed_10', 'std_speed_10'
        ]

        # 훈련 및 테스트 데이터 준비
        X_train = train_data[feature_cols]
        y_train = train_data['Residual']

        X_test = test_data[feature_cols]
        y_test = test_data['Residual']

        # 훈련 데이터를 추가로 훈련/검증 세트로 분할
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        if params is None:
            print("No parameters provided. Please provide SVR parameters.")
            return
        else:
            best_params = params

        model = train_model(X_train, y_train, best_params)

        # 예측 수행
        train_data['y_pred'] = model.predict(X_train)
        test_data['y_pred'] = model.predict(X_test)

        if fold_num == 2 and plot == True:
            # 컴포지트 그림 생성
            plot_composite_contour(
                X_train=X_train[['speed', 'acceleration']].values,
                y_pred_train=train_data['y_pred'].values,
                X_test=X_test[['speed', 'acceleration']].values,
                y_pred_test1=test_data['y_pred'].values,
                y_pred_test2=(test_data['Residual'] - test_data['y_pred']).values,
                scaler=scaler,
                selected_car=selected_car,
                terminology1=f'{selected_car} : Train Residual',
                terminology2=f'{selected_car} : Residual[1]',
                terminology3=f'{selected_car} : Residual[2]',
                num_grids=30,
                min_count=10,
                save_directory=r"C:\Users\BSL\Desktop\Figures"
            )

        # 트립별로 적분 수행
        train_trip_groups = train_data.groupby('trip_id')
        test_trip_groups = test_data.groupby('trip_id')

        hybrid_integrals_train, data_integrals_train = [], []
        for _, group in train_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_train.append(hybrid_integral)
            data_integrals_train.append(data_integral)

        # 훈련 데이터의 MAPE 및 RRMSE 계산
        mape_train = calculate_mape(np.array(data_integrals_train), np.array(hybrid_integrals_train))
        rrmse_train = calculate_rrmse(np.array(data_integrals_train), np.array(hybrid_integrals_train))

        # 테스트 데이터의 적분 수행
        hybrid_integrals_test, data_integrals_test = [], []
        for _, group in test_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_test.append(hybrid_integral)
            data_integrals_test.append(data_integral)

        # 테스트 데이터의 MAPE 및 RRMSE 계산
        mape_test = calculate_mape(np.array(data_integrals_test), np.array(hybrid_integrals_test))
        rrmse_test = calculate_rrmse(np.array(data_integrals_test), np.array(hybrid_integrals_test))

        # RMSE 계산
        rmse = calculate_rmse(
            (y_test + test_data['Power_phys']),
            (test_data['y_pred'] + test_data['Power_phys'])
        )

        # 결과 저장
        results.append({
            'fold': fold_num,
            'rmse': rmse,
            'test_rrmse': rrmse_test,
            'test_mape': mape_test,
            'best_params': best_params
        })
        models.append(model)

        # 폴드 결과 출력
        print(f"--- Fold {fold_num} Results ---")
        print(f"Params: {best_params}")
        print(f"RMSE : {rmse:.2f}")
        print(f"Train - MAPE: {mape_train:.2f}%, RRMSE: {rrmse_train:.4f}")
        print(f"Test - MAPE: {mape_test:.2f}%, RRMSE: {rrmse_test:.4f}")
        print("---------------------------\n")

    # 모든 폴드가 완료된 후 최적의 모델 선택
    if len(results) == kf.get_n_splits():
        # 모든 폴드의 RMSE 값을 추출
        rmse_values = [result['rmse'] for result in results]

        # RMSE의 중앙값 계산
        median_rmse = np.median(rmse_values)

        # 중앙값과 가장 가까운 RMSE 값을 가진 폴드의 인덱스 찾기
        closest_index = np.argmin(np.abs(np.array(rmse_values) - median_rmse))

        # 해당 인덱스의 모델을 best_model로 선택
        best_model = models[closest_index]

        # 해당 폴드의 하이퍼파라미터를 best_params_overall로 설정
        best_params_overall = results[closest_index]['best_params']

        # 선택된 폴드의 정보를 출력
        selected_fold = results[closest_index]['fold']
        print(f"Selected Fold {selected_fold} as Best Model with RMSE: {rmse_values[closest_index]:.4f}")
    else:
        best_model = None
        print("No models available to select as best_model.")

    return results, scaler, best_params_overall