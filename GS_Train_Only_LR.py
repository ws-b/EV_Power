import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from GS_Functions import calculate_rmse, calculate_mape
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.integrate import cumulative_trapezoid

# =========================
# 데이터 처리 함수들
# =========================

def process_single_file(file):
    """
    단일 CSV 파일을 읽어서 필요한 컬럼만 리턴합니다.
    이번 버전에서는 'Power_phys', 'Residual'을 전혀 사용하지 않습니다.
    'Power_data'가 존재하는지 체크한 뒤, 해당 컬럼을 포함하여 반환합니다.
    """
    try:
        data = pd.read_csv(file)
        if 'Power_data' in data.columns:
            # 필요한 열만 선택
            return data[['time', 'speed', 'acceleration', 'ext_temp', 'Power_data']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None


def process_files(files, scaler=None):
    """
    여러 CSV 파일을 병렬로 처리하고, 롤링 통계량을 계산한 뒤 피처를 스케일링합니다.
    """
    # 스케일 범위를 정의 (필요에 따라 변경 가능)
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6  # km/h -> m/s
    ACCELERATION_MIN = -15  # m/s^2
    ACCELERATION_MAX = 9    # m/s^2
    TEMP_MIN = -30
    TEMP_MAX = 50

    # 스케일링 대상 피처들
    feature_cols = [
        'speed', 'acceleration', 'ext_temp',
        'mean_accel_10', 'std_accel_10',
        'mean_speed_10', 'std_speed_10'
    ]

    df_list = []
    # files를 나열하면서 index, file을 가져옴
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
                    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

                    # trip_id를 index로 지정 (파일별 구분)
                    data['trip_id'] = idx

                    # ---------------------
                    # 롤링 통계량 계산 (window=5)
                    # ---------------------
                    data['mean_accel_10'] = (
                        data['acceleration'].rolling(window=5).mean().bfill()
                    )
                    data['std_accel_10'] = (
                        data['acceleration'].rolling(window=5).std().bfill()
                    )
                    data['mean_speed_10'] = (
                        data['speed'].rolling(window=5).mean().bfill()
                    )
                    data['std_speed_10'] = (
                        data['speed'].rolling(window=5).std().bfill()
                    )

                    df_list.append(data)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    full_data = pd.concat(df_list, ignore_index=True)

    # -----------------------------------
    # MinMaxScaler 초기화 (처음 한 번만)
    # -----------------------------------
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 최솟값과 최댓값으로 한번에 fit
        scaler.fit(
            pd.DataFrame([
                [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN, 0, 0, 0, 0],
                [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX, 1, 1, 1, 1]
            ], columns=feature_cols)
        )

    # 모든 특징(feature_cols)에 스케일링 적용
    full_data[feature_cols] = scaler.transform(full_data[feature_cols])

    return full_data, scaler


def integrate_and_compare(trip_data):
    """
    트립 데이터에서 예측값(y_pred)과 실제값(Power_data)를 시간축에 대해 적분 비교합니다.
    """
    # time으로 정렬
    trip_data = trip_data.sort_values(by='time')

    # time을 초 단위로 변환
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # 예측값 적분
    pred_cum = cumulative_trapezoid(trip_data['y_pred'].values, time_seconds, initial=0)
    pred_integral = pred_cum[-1]

    # 실제값(Power_data) 적분
    data_cum = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)
    data_integral = data_cum[-1]

    return pred_integral, data_integral


def train_model_linear_regression(X_train, y_train):
    """
    Linear Regression 모델을 훈련시킵니다.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# =========================
# 수정된 함수 (Median Fold)
# =========================

def train_validate_test(vehicle_files, selected_car):
    """
    1) 전체 파일 중 20%를 test로, 80%를 train으로 split
    2) train 부분만 5-Fold 교차 검증 진행 (Linear Regression)
    3) 5-Fold 중 Validation RMSE 값들의 중앙값(median)에 가장 가까운 fold를 best model로 선정
    4) 선정된 best model로 test 세트 최종 평가(MAPE, RMSE)
    """
    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    # --------------------------------------------------------------------------------
    # (1) Train/Test Split (파일 기준)
    # --------------------------------------------------------------------------------
    all_files = vehicle_files[selected_car]
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

    if len(train_files) == 0:
        raise ValueError("No training files were found after splitting. Check file list or split ratio.")
    if len(test_files) == 0:
        raise ValueError("No test files were found after splitting. Check file list or split ratio.")

    # --------------------------------------------------------------------------------
    # (2) 5-Fold 교차 검증 (train_files 기준)
    # --------------------------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_files = np.array(train_files)  # numpy array로 변환

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

        # ---------------------------
        # Fold train/val 데이터 처리
        # ---------------------------
        train_data, scaler = process_files(fold_train_files)
        val_data, _ = process_files(fold_val_files, scaler=scaler)

        # ---------------------------
        # 모델 학습
        # ---------------------------
        X_train = train_data[feature_cols]
        y_train = train_data['Power_data']  # 이번엔 Power_data가 레이블

        X_val = val_data[feature_cols]
        y_val = val_data['Power_data']

        model = train_model_linear_regression(X_train, y_train)

        # Validation 예측
        val_data['y_pred'] = model.predict(X_val)

        # ---------------------------
        # Validation 적분 기반 MAPE, RMSE
        # ---------------------------
        val_trip_groups = val_data.groupby('trip_id')

        pred_integrals_val, data_integrals_val = [], []
        for _, group in val_trip_groups:
            pred_integral, data_integral = integrate_and_compare(group)
            pred_integrals_val.append(pred_integral)
            data_integrals_val.append(data_integral)

        # 적분 기반 MAPE
        mape_val = calculate_mape(
            np.array(data_integrals_val),
            np.array(pred_integrals_val)
        )

        # 시계열 RMSE (단순 시계열)
        rmse_val = calculate_rmse(
            val_data['Power_data'],
            val_data['y_pred']
        )

        fold_results.append({
            'fold': fold_num,
            'rmse': rmse_val,
            'mape': mape_val
        })
        fold_models.append(model)
        fold_scalers.append(scaler)

        print(f"[Fold {fold_num}] Validation RMSE = {rmse_val:.4f}, MAPE = {mape_val:.2f}%")

    # --------------------------------------------------------------------------------
    # (3) 교차 검증 결과 중 RMSE의 "중앙값(median)"에 가장 가까운 fold를 선택 => best model
    # --------------------------------------------------------------------------------
    val_rmse_values = [res['rmse'] for res in fold_results]
    median_rmse = np.median(val_rmse_values)
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
    # --------------------------------------------------------------------------------
    test_data, _ = process_files(test_files, scaler=best_scaler)

    X_test = test_data[feature_cols]
    y_test = test_data['Power_data']

    test_data['y_pred'] = best_model.predict(X_test)

    # 테스트 적분 기반 비교
    test_trip_groups = test_data.groupby('trip_id')
    pred_integrals_test, data_integrals_test = [], []
    for _, group in test_trip_groups:
        pred_integral, data_integral = integrate_and_compare(group)
        pred_integrals_test.append(pred_integral)
        data_integrals_test.append(data_integral)

    mape_test = calculate_mape(
        np.array(data_integrals_test),
        np.array(pred_integrals_test)
    )
    rmse_test = calculate_rmse(
        test_data['Power_data'],
        test_data['y_pred']
    )

    print(f"\n[Test Set Evaluation using Best Model (Fold {best_fold})]")
    print(f"  RMSE : {rmse_test:.4f}")
    print(f"  MAPE : {mape_test:.2f}%")
    print("------------------------------------")

    results = [{
        'fold_results': fold_results,
        'best_fold': best_fold,
        'best_model': best_model,
        'test_rmse': rmse_test,
        'test_mape': mape_test
    }]

    return results, best_scaler
