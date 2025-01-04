import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from GS_Functions import calculate_rmse, calculate_mape
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.integrate import cumulative_trapezoid


# =========================
# 데이터 처리 함수들 (Data Processing Functions)
# =========================

def process_single_file(file):
    """
    단일 CSV 파일을 처리하여 잔차를 계산하고 관련 열을 선택합니다.
    (EN) Process a single CSV file to compute residuals and select relevant columns.
    """
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            # 'Residual' = 'Power_data' - 'Power_phys'
            # (EN) Calculate the residual by subtracting 'Power_phys' from 'Power_data'
            data['Residual'] = data['Power_data'] - data['Power_phys']
            return data[['time', 'speed', 'acceleration', 'ext_temp',
                         'Residual', 'Power_phys', 'Power_data']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None


def process_files(files, scaler=None):
    """
    여러 CSV 파일을 병렬로 처리하고, 롤링 통계량을 계산하며 특징을 스케일링합니다.
    (EN) Process multiple CSV files in parallel, calculate rolling statistics,
         and apply feature scaling.
    """
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6  # km/h -> m/s
    ACCELERATION_MIN = -15  # m/s^2
    ACCELERATION_MAX = 9    # m/s^2
    TEMP_MIN = -30
    TEMP_MAX = 50

    feature_cols = [
        'speed', 'acceleration', 'ext_temp',
        'mean_accel_10', 'std_accel_10',
        'mean_speed_10', 'std_speed_10'
    ]

    df_list = []
    # files를 나열하면서 index, file을 가져옴
    # (EN) Iterate over the files, retrieving index and file
    with ProcessPoolExecutor() as executor:
        future_to_info = {
            executor.submit(process_single_file, file): (idx, file)
            for idx, file in enumerate(files)
        }

        for future in as_completed(future_to_info):
            idx, file = future_to_info[future]
            try:
                data = future.result()
                if data is not None:
                    # 'time' 열을 datetime으로 변환
                    # (EN) Convert the 'time' column to datetime format
                    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

                    # trip_id를 index로 직접 지정
                    # (EN) Assign the trip ID explicitly based on the file index
                    data['trip_id'] = idx

                    # 윈도우 크기 5로 롤링 통계량 계산
                    # (EN) Calculate rolling statistics with a window of size 5
                    data['mean_accel_10'] = data['acceleration'].rolling(window=5).mean().bfill()
                    data['std_accel_10'] = data['acceleration'].rolling(window=5).std().bfill()
                    data['mean_speed_10'] = data['speed'].rolling(window=5).mean().bfill()
                    data['std_speed_10'] = data['speed'].rolling(window=5).std().bfill()

                    df_list.append(data)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    # 여러 개의 DataFrame을 하나로 합침
    # (EN) Concatenate all DataFrames into one
    full_data = pd.concat(df_list, ignore_index=True)

    # 스케일러 초기화
    # (EN) Initialize the scaler
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame([
            [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN, 0, 0, 0, 0],
            [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX, 1, 1, 1, 1]
        ], columns=feature_cols))

    # 모든 특징에 스케일링 적용
    # (EN) Apply scaling to all features
    full_data[feature_cols] = scaler.transform(full_data[feature_cols])

    return full_data, scaler


def integrate_and_compare(trip_data):
    """
    트립 데이터에서 'Power_hybrid'와 'Power_data'를 시간에 따라 적분합니다.
    (EN) Perform time-based integration on 'Power_hybrid' and 'Power_data' for trip data.
    """
    # 'time'으로 정렬
    # (EN) Sort by 'time'
    trip_data = trip_data.sort_values(by='time')

    # 'time'을 초 단위로 변환
    # (EN) Convert 'time' to seconds
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # Power_hybrid = Power_phys + y_pred
    # (EN) Calculate 'Power_hybrid' as the sum of 'Power_phys' and 'y_pred'
    trip_data['Power_hybrid'] = trip_data['Power_phys'] + trip_data['y_pred']
    hybrid_cum_integral = cumulative_trapezoid(trip_data['Power_hybrid'].values, time_seconds, initial=0)
    hybrid_integral = hybrid_cum_integral[-1]

    # Power_data 적분
    # (EN) Integrate 'Power_data'
    data_cum_integral = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)
    data_integral = data_cum_integral[-1]

    return hybrid_integral, data_integral


def train_model_linear_regression(X_train, y_train):
    """
    Linear Regression 모델을 훈련시킵니다.
    (EN) Train a Linear Regression model using the given training data.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# =========================
# 수정된 함수 (Median Fold)
# (EN) Modified function (Median Fold)
# =========================

def train_validate_test(vehicle_files):
    """
    1) 전체 파일 중 20%를 test로, 80%를 train으로 split
       (EN) Split 20% of the files for testing and 80% for training
    2) train 부분만 5-Fold 교차 검증 진행 (Linear Regression)
       (EN) Perform 5-Fold cross-validation on the training portion (Linear Regression)
    3) 5-Fold 중 Validation RMSE 값들의 중앙값(median)에 가장 가까운 fold를 best model로 선정
       (EN) Among the 5-Fold results, select the fold whose Validation RMSE is closest to the median as the best model
    4) 선정된 best model로 test 세트 최종 평가(MAPE, RMSE)
       (EN) Finally, evaluate on the test set (MAPE, RMSE) using the selected best model
    """
    # vehicle_files가 비어있으면 예외 처리
    # (EN) Raise an exception if 'vehicle_files' is empty
    if not vehicle_files:
        raise ValueError("No files provided. The 'vehicle_files' list is empty.")

    # --------------------------------------------------------------------------------
    # (1) Train/Test Split (파일 기준)
    # (EN) Step (1): Split the dataset into train and test by file
    # --------------------------------------------------------------------------------
    all_files = vehicle_files
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

    if len(train_files) == 0:
        raise ValueError("No training files were found after splitting. Check file list or split ratio.")
    if len(test_files) == 0:
        raise ValueError("No test files were found after splitting. Check file list or split ratio.")

    # --------------------------------------------------------------------------------
    # (2) 5-Fold 교차 검증 (train_files 기준)
    # (EN) Step (2): 5-Fold cross-validation on train_files
    # --------------------------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_files = np.array(train_files)  # numpy array로 변환
    # (EN) Convert train_files to a numpy array

    fold_results = []
    fold_models = []
    fold_scalers = []

    feature_cols = [
        'speed', 'acceleration', 'ext_temp',
        'mean_accel_10', 'std_accel_10',
        'mean_speed_10', 'std_speed_10'
    ]

    for fold_num, (fold_train_idx, fold_val_idx) in enumerate(kf.split(train_files), start=1):
        fold_train_files = train_files[fold_train_idx]
        fold_val_files = train_files[fold_val_idx]

        # --------------------
        # Fold train/val 데이터 처리
        # (EN) Process data for fold train/val
        # --------------------
        train_data, scaler = process_files(fold_train_files)
        val_data, _ = process_files(fold_val_files, scaler=scaler)

        # 모델 학습
        # (EN) Train the model
        X_train = train_data[feature_cols]
        y_train = train_data['Residual']

        X_val = val_data[feature_cols]
        y_val = val_data['Residual']

        model = train_model_linear_regression(X_train, y_train)

        # Validation 예측
        # (EN) Make predictions on the validation set
        val_data['y_pred'] = model.predict(X_val)

        # --------------------
        # Validation 적분 기반 RMSE, MAPE 계산
        # (EN) Compute RMSE and MAPE using integration results
        # --------------------
        # 적분은 trip_id 단위로 수행
        # (EN) Perform integration by trip_id
        val_trip_groups = val_data.groupby('trip_id')

        hybrid_integrals_val, data_integrals_val = [], []
        for _, group in val_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_val.append(hybrid_integral)
            data_integrals_val.append(data_integral)

        # 적분 기반 MAPE
        # (EN) MAPE based on integrated values
        mape_val = calculate_mape(np.array(data_integrals_val),
                                  np.array(hybrid_integrals_val))

        # 시계열 RMSE (단순 시계열로 계산)
        # (EN) Time-series RMSE (simple time-series calculation)
        rmse_val = calculate_rmse(
            (y_val + val_data['Power_phys']),
            (val_data['y_pred'] + val_data['Power_phys'])
        )

        fold_results.append({
            'fold': fold_num,
            'rmse': rmse_val,   # Median Fold 판단에 사용
            # (EN) Used to determine the Median Fold
            'mape': mape_val
        })
        fold_models.append(model)
        fold_scalers.append(scaler)

        print(f"[Fold {fold_num}] Validation RMSE = {rmse_val:.4f}, MAPE = {mape_val:.2f}%")

    # --------------------------------------------------------------------------------
    # (3) 교차 검증 종료 후, Validation RMSE의 "중앙값(median)"에 가장 가까운 fold를 선택 => best model
    # (EN) After cross-validation, pick the fold whose Validation RMSE is closest to the median => best model
    # --------------------------------------------------------------------------------
    val_rmse_values = [res['rmse'] for res in fold_results]
    median_rmse = np.median(val_rmse_values)
    # median과의 차이가 가장 작은 fold 인덱스 선택
    # (EN) Choose the index of the fold that is closest to the median
    closest_index = np.argmin(np.abs(np.array(val_rmse_values) - median_rmse))

    best_model_info = fold_results[closest_index]
    best_model = fold_models[closest_index]
    best_scaler = fold_scalers[closest_index]
    best_fold = best_model_info['fold']

    print(f"\n[Best Model Selection]")
    print(f"  => Fold {best_fold} selected.")
    print(f"     (Validation RMSE: {best_model_info['rmse']:.4f}, MAPE: {best_model_info['mape']:.2f}%)")

    # --------------------------------------------------------------------------------
    # (4) 선정된 best model로 Test 세트 최종 평가
    # (EN) Evaluate the selected best model on the test set
    # --------------------------------------------------------------------------------
    test_data, _ = process_files(test_files, scaler=best_scaler)

    X_test = test_data[feature_cols]
    y_test = test_data['Residual']

    test_data['y_pred'] = best_model.predict(X_test)

    # Test 적분 기반 MAPE, 시계열 RMSE
    # (EN) MAPE and RMSE on the test set based on integration and time-series data
    test_trip_groups = test_data.groupby('trip_id')
    hybrid_integrals_test, data_integrals_test = [], []
    for _, group in test_trip_groups:
        hybrid_integral, data_integral = integrate_and_compare(group)
        hybrid_integrals_test.append(hybrid_integral)
        data_integrals_test.append(data_integral)

    mape_test = calculate_mape(np.array(data_integrals_test),
                               np.array(hybrid_integrals_test))
    rmse_test = calculate_rmse(
        (y_test + test_data['Power_phys']),
        (test_data['y_pred'] + test_data['Power_phys'])
    )

    print(f"\n[Test Set Evaluation using Best Model (Fold {best_fold})]")
    print(f"  RMSE : {rmse_test:.4f}")
    print(f"  MAPE : {mape_test:.2f}%")
    print("------------------------------------")

    results = []
    results.append({
        'fold_results': fold_results,   # 각 fold별 {'fold', 'rmse', 'mape'}
        # (EN) For each fold: {'fold', 'rmse', 'mape'}
        'best_fold': best_fold,
        'best_model': best_model,
        'test_rmse': rmse_test,
        'test_mape': mape_test
    })

    return results, best_scaler
