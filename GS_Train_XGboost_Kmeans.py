import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from GS_Functions import calculate_rrmse, calculate_rmse, calculate_mape
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_file(file):
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            # 'time' 열을 datetime 형식으로 변환
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

            # 'jerk' 계산 (가속도의 변화율)
            data['jerk'] = data['acceleration'].diff().fillna(0)

            return data[['time', 'speed', 'acceleration', 'jerk', 'ext_temp', 'Residual', 'Power_phys', 'Power_data']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None

def process_files(files, scaler=None, residual_scaler=None, kmeans=None, n_clusters=5):
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6  # 230km/h 를 m/s 로 변환
    ACCELERATION_MIN = -15  # m/s^2
    ACCELERATION_MAX = 9    # m/s^2
    JERK_MIN = -10          # 가속도의 최소 변화율
    JERK_MAX = 10           # 가속도의 최대 변화율
    TEMP_MIN = -30
    TEMP_MAX = 50

    feature_cols = ['speed', 'acceleration', 'jerk', 'ext_temp',
                    'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10']

    df_list = []
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                data = future.result()
                if data is not None:
                    # Trip 구분을 위해 각 데이터에 파일 인덱스(Trip ID) 추가
                    data['trip_id'] = files.index(file)

                    # 롤링 통계량 계산
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
            [SPEED_MIN, ACCELERATION_MIN, JERK_MIN, TEMP_MIN, 0, 0, 0, 0],
            [SPEED_MAX, ACCELERATION_MAX, JERK_MAX, TEMP_MAX, 1, 1, 1, 1]
        ], columns=feature_cols))

    # 모든 피처에 대해 스케일링 적용
    full_data[feature_cols] = scaler.transform(full_data[feature_cols])

    # Residual 스케일링
    if residual_scaler is None:
        residual_scaler = StandardScaler()
        residual_scaler.fit(full_data[['Residual']])
    full_data['Residual_scaled'] = residual_scaler.transform(full_data[['Residual']])

    # 클러스터링에 사용할 특징들
    clustering_features = ['speed', 'acceleration', 'jerk', 'Residual_scaled']

    # K-Means 클러스터링 적용
    if kmeans is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(full_data[clustering_features])

    # 클러스터 레이블 추가
    full_data['cluster_label'] = kmeans.predict(full_data[clustering_features])

    return full_data, scaler, residual_scaler, kmeans

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

def cross_validate(vehicle_files, selected_car, precomputed_lambda=None, plot=None, save_dir="models", n_clusters=5):
    model_name = "XGB"

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []
    best_model = None
    scaler = None
    residual_scaler = None
    kmeans = None

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]

    for fold_num, (train_index, test_index) in enumerate(kf.split(files), 1):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        # Train set과 test set을 처리
        train_data, scaler, residual_scaler, kmeans = process_files(train_files, scaler, residual_scaler, kmeans, n_clusters)
        test_data, _, _, _ = process_files(test_files, scaler, residual_scaler, kmeans, n_clusters)

        # 각 Trip별로 그룹화하여 적분 수행
        train_trip_groups = train_data.groupby('trip_id')
        test_trip_groups = test_data.groupby('trip_id')

        feature_cols = ['speed', 'acceleration', 'jerk', 'ext_temp',
                        'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10', 'cluster_label']
        # 학습에 사용할 데이터 준비
        X_train = train_data[feature_cols].to_numpy()
        y_train = train_data['Residual'].to_numpy()

        X_test = test_data[feature_cols].to_numpy()
        y_test = test_data['Residual'].to_numpy()

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # 모델 학습
        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': ['rmse'],
            'lambda': precomputed_lambda if precomputed_lambda else 1.0,
            'eta': 0.3
        }

        evals = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(params, dtrain, num_boost_round=150, evals=evals)
        train_data['y_pred'] = model.predict(dtrain)
        test_data['y_pred'] = model.predict(dtest)

        # Train set에서 Trip별로 적분을 계산하고 MAPE 및 RRMSE 계산
        hybrid_integrals_train, data_integrals_train = [], []
        for _, group in train_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_train.append(hybrid_integral)
            data_integrals_train.append(data_integral)

        # MAPE 및 RRMSE 계산
        mape_train = calculate_mape(np.array(data_integrals_train), np.array(hybrid_integrals_train))
        rrmse_train = calculate_rrmse(np.array(data_integrals_train), np.array(hybrid_integrals_train))

        # Test set에서도 동일한 방식으로 MAPE 및 RRMSE 계산
        hybrid_integrals_test, data_integrals_test = [], []
        for _, group in test_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_test.append(hybrid_integral)
            data_integrals_test.append(data_integral)

        # MAPE 및 RRMSE 계산
        mape_test = calculate_mape(np.array(data_integrals_test), np.array(hybrid_integrals_test))
        rrmse_test = calculate_rrmse(np.array(data_integrals_test), np.array(hybrid_integrals_test))

        rmse = calculate_rmse((y_test + test_data['Power_phys']), (test_data['y_pred'] + test_data['Power_phys']))

        results.append((fold_num, rmse, rrmse_train, mape_train, rrmse_test, mape_test))
        models.append(model)

        # 각 Fold별 결과 출력
        print(f"Vehicle: {selected_car}, Fold: {fold_num}, n_clusters: {n_clusters}")
        print(f"Train - MAPE: {mape_train:.2f}, RRMSE: {rrmse_train:.2f}")
        print(f"Test - MAPE: {mape_test:.2f}, RRMSE: {rrmse_test:.2f}")

    # 최종 모델 선택 (Test set에서 MAPE 기준으로 중간값에 해당하는 모델)
    median_mape = np.median([result[5] for result in results])
    median_index = np.argmin([abs(result[5] - median_mape) for result in results])
    best_model = models[median_index]

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 최적 모델 저장
        if best_model:
            model_file = os.path.join(save_dir, f"{model_name}_best_model_{selected_car}_k{n_clusters}.json")
            best_model.save_model(model_file)
            print(f"Best model for {selected_car} saved with MAPE: {median_mape}")

        # 스케일러 저장
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler_{selected_car}_k{n_clusters}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at {scaler_path}")

        # Residual 스케일러 저장
        residual_scaler_path = os.path.join(save_dir, f'{model_name}_residual_scaler_{selected_car}_k{n_clusters}.pkl')
        with open(residual_scaler_path, 'wb') as f:
            pickle.dump(residual_scaler, f)
        print(f"Residual scaler saved at {residual_scaler_path}")

        # K-Means 모델 저장
        kmeans_path = os.path.join(save_dir, f'{model_name}_kmeans_{selected_car}_k{n_clusters}.pkl')
        with open(kmeans_path, 'wb') as f:
            pickle.dump(kmeans, f)
        print(f"K-Means model saved at {kmeans_path}")

    return results, scaler, residual_scaler, kmeans

def process_file_with_trained_model(file, model, scaler, residual_scaler, kmeans):
    try:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns and 'Power_phys' in data.columns:
            # 'time' 열을 datetime 형식으로 변환
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

            # 'jerk' 계산
            data['jerk'] = data['acceleration'].diff().fillna(0)

            # 롤링 통계량 계산
            data['mean_accel_10'] = data['acceleration'].rolling(window=5).mean().bfill()
            data['std_accel_10'] = data['acceleration'].rolling(window=5).std().bfill()
            data['mean_speed_10'] = data['speed'].rolling(window=5).mean().bfill()
            data['std_speed_10'] = data['speed'].rolling(window=5).std().bfill()

            # Residual 계산 및 스케일링
            data['Residual'] = data['Power_data'] - data['Power_phys']
            data['Residual_scaled'] = residual_scaler.transform(data[['Residual']])

            feature_cols = ['speed', 'acceleration', 'jerk', 'ext_temp',
                            'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10']

            # 필요한 피처들에 대해 스케일링 적용
            features = data[feature_cols]
            features_scaled = scaler.transform(features)

            # 클러스터링에 사용할 특징들
            clustering_features = np.hstack([features_scaled[:, :3], data['Residual_scaled'].values.reshape(-1, 1)])

            # 클러스터 레이블 예측
            data['cluster_label'] = kmeans.predict(clustering_features)

            # 최종 피처에 클러스터 레이블 추가
            features_scaled = np.hstack([features_scaled, data['cluster_label'].values.reshape(-1, 1)])

            # Predict the residual using the trained model
            dtest = xgb.DMatrix(features_scaled)
            predicted_residual = model.predict(dtest)

            # Calculate the hybrid power
            data['Power_hybrid'] = predicted_residual + data['Power_phys']
            save_column = ['time', 'speed', 'acceleration', 'ext_temp', 'Power_data', 'Power_phys',
                           'Power_hybrid']

            # Save the updated file
            data.to_csv(file, columns=save_column, index=False)

            print(f"Processed file {file}")
        else:
            print(f"File {file} does not contain required columns 'speed', 'acceleration', or 'Power_phys'.")
    except Exception as e:
        print(f"Error processing file {file}: {e}")

def add_predicted_power_column(files, model_path, scaler_path, residual_scaler_path, kmeans_path):
    try:
        # Load the trained model
        model = xgb.Booster()
        model.load_model(model_path)

        # Load the scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Load the residual scaler
        with open(residual_scaler_path, 'rb') as f:
            residual_scaler = pickle.load(f)

        # Load the K-Means model
        with open(kmeans_path, 'rb') as f:
            kmeans = pickle.load(f)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_with_trained_model, file, model, scaler, residual_scaler, kmeans) for file in files]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing file: {e}")
