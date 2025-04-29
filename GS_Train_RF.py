# GS_Train_RF.py
import os
import pandas as pd
import platform
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.integrate import cumulative_trapezoid
from GS_Functions import calculate_mape
# from GS_plot import plot_composite_contour # RF는 SHAP이나 특정 플롯이 적합하지 않을 수 있음
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# ----------------------------
# 전역 변수 / 상수 정의 (XGBoost/DNN과 동일)
# ----------------------------
SPEED_MIN = 0 / 3.6
SPEED_MAX = 230 / 3.6
ACCELERATION_MIN = -15
ACCELERATION_MAX = 9
TEMP_MIN = -30
TEMP_MAX = 50
ACCEL_STD_MAX = 10
SPEED_STD_MAX = 30

window_sizes = [5]

def generate_feature_columns():
    feature_cols = ['speed', 'acceleration', 'ext_temp']
    for w in window_sizes:
        time_window = w * 2
        feature_cols.extend([
            f'mean_accel_{time_window}',
            f'std_accel_{time_window}',
            f'mean_speed_{time_window}',
            f'std_speed_{time_window}'
        ])
    return feature_cols

FEATURE_COLS = generate_feature_columns()

# ----------------------------
# 데이터 처리 함수 (XGBoost/DNN과 동일)
# ----------------------------
def process_single_file(file, trip_id):
    """ 파일 하나를 읽어 rolling feature 계산 후 반환 (XGBoost/DNN과 동일) """
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

            for w in window_sizes:
                time_window = w * 2
                data[f'mean_accel_{time_window}'] = data['acceleration'].rolling(window=w, min_periods=1).mean().bfill()
                data[f'std_accel_{time_window}'] = data['acceleration'].rolling(window=w, min_periods=1).std().bfill().fillna(0)
                data[f'mean_speed_{time_window}'] = data['speed'].rolling(window=w, min_periods=1).mean().bfill()
                data[f'std_speed_{time_window}'] = data['speed'].rolling(window=w, min_periods=1).std().bfill().fillna(0)

            data['trip_id'] = trip_id
            # 필요한 컬럼만 선택 (메모리 효율성)
            required_cols = FEATURE_COLS + ['Residual', 'trip_id', 'time', 'Power_phys', 'Power_data']
            if all(col in data.columns for col in required_cols):
                 return data[required_cols]
            else:
                 # print(f"Warning: Missing required columns in file {file} after processing. Skipping.")
                 return None
        else:
            # print(f"Warning: 'Power_phys' or 'Power_data' missing in {file}. Skipping.")
            return None
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None

def scale_data(df, scaler=None):
    """ FEATURE_COLS에 대해 MinMaxScaling (XGBoost/DNN과 동일) """
    if scaler is None:
        min_vals = [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN]
        max_vals = [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX]
        window_val_min = [ACCELERATION_MIN, 0, SPEED_MIN, 0]
        window_val_max = [ACCELERATION_MAX, ACCEL_STD_MAX, SPEED_MAX, SPEED_STD_MAX]

        for w in window_sizes:
            min_vals.extend(window_val_min)
            max_vals.extend(window_val_max)

        dummy_df_min = pd.DataFrame([min_vals], columns=FEATURE_COLS)
        dummy_df_max = pd.DataFrame([max_vals], columns=FEATURE_COLS)
        dummy_df = pd.concat([dummy_df_min, dummy_df_max], ignore_index=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(dummy_df)

    df_to_scale = df[FEATURE_COLS].copy()
    scaled_values = scaler.transform(df_to_scale)
    df[FEATURE_COLS] = scaled_values
    return df, scaler

def integrate_and_compare(trip_data):
    """ Trip 데이터 에너지 적분 및 비교 (XGBoost/DNN과 동일, 'rf_pred' 사용 가정) """
    trip_data = trip_data.sort_values(by='time')
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # RF 예측값을 Residual 예측으로 사용
    trip_data['Power_hybrid'] = trip_data['Power_phys'] + trip_data['rf_pred'] # 'rf_pred' 컬럼 필요

    hybrid_cum_integral = cumulative_trapezoid(trip_data['Power_hybrid'].values, time_seconds, initial=0)
    hybrid_integral = hybrid_cum_integral[-1] if len(hybrid_cum_integral) > 0 else 0

    data_cum_integral = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)
    data_integral = data_cum_integral[-1] if len(data_cum_integral) > 0 else 0

    return hybrid_integral, data_integral

# ----------------------------
# Random Forest 워크플로우 함수
# ----------------------------
def run_rf_workflow(vehicle_files, selected_car, plot=False, save_dir="models_rf"):
    """
    Scikit-learn Random Forest 모델 학습 및 평가 워크플로우.
    """
    start_workflow_time = time.time()
    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return [], None

    files = vehicle_files[selected_car]
    print(f"Starting Random Forest workflow for {selected_car} with {len(files)} files...")

    # 1. 파일 단위 Train/Test 분할 (8:2)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    print(f"Split: {len(train_files)} train files, {len(test_files)} test files.")

    # 2. Train 데이터 처리
    print("Processing training files...")
    train_data_list = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, f, trip_id=i) for i, f in enumerate(train_files)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                train_data_list.append(result)

    if not train_data_list:
        print(f"Error: No valid training data could be processed for {selected_car}.")
        return [], None

    train_data = pd.concat(train_data_list, ignore_index=True)
    print(f"Training data shape: {train_data.shape}")

    # 3. Scaler Fit (RF는 스케일링에 덜 민감하지만, 일관성을 위해 적용)
    print("Fitting scaler...")
    train_data_scaled, scaler = scale_data(train_data.copy())
    X_train = train_data_scaled[FEATURE_COLS] # 스케일링된 특성 사용
    y_train = train_data_scaled['Residual']   # 타겟 변수

    # 4. Test 데이터 처리 및 스케일링
    print("Processing test files...")
    test_data_list = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, f, trip_id=1000 + j) for j, f in enumerate(test_files)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                test_data_list.append(result)

    if not test_data_list:
        print(f"Warning: No valid test data processed for {selected_car}. Cannot evaluate.")
        return [], scaler

    test_data = pd.concat(test_data_list, ignore_index=True)
    print(f"Test data shape: {test_data.shape}")

    # Test 데이터 스케일링 (Fit된 scaler 사용)
    test_data_scaled, _ = scale_data(test_data.copy(), scaler)
    X_test = test_data_scaled[FEATURE_COLS]
    y_test = test_data_scaled['Residual'] # 실제 Residual 값 (Ground Truth)

    # -----------------------
    # 5. Random Forest 모델 학습 (하이퍼파라미터 튜닝 생략, 기본값 사용)
    # -----------------------
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=100,  # 트리의 개수 (기본값, 필요시 조정)
        random_state=42,   # 재현성을 위한 시드 고정
        n_jobs=-1,         # 사용 가능한 모든 CPU 코어 활용
        max_depth=None,      # 트리의 최대 깊이 (제한 없음, 필요시 조정)
        min_samples_split=2, # 노드를 분할하기 위한 최소 샘플 수 (기본값)
        min_samples_leaf=1   # 리프 노드가 되기 위한 최소 샘플 수 (기본값)
        # oob_score=True    # Out-of-Bag 샘플로 검증 점수 계산 (선택 사항)
    )

    train_start_time = time.time()
    rf_model.fit(X_train, y_train) # X_train, y_train은 DataFrame/Series 또는 Numpy 배열
    train_end_time = time.time()
    print(f"Random Forest training finished in {train_end_time - train_start_time:.2f} seconds.")

    # -----------------------
    # 6. Test Set 평가
    # -----------------------
    print("Evaluating model on the test set...")
    y_pred_test_rf = rf_model.predict(X_test)

    # Test RMSE (Residual 기준)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))
    print(f"Test RMSE (on Residual): {test_rmse:.4f}")

    # Test MAPE (Energy 기준)
    # 예측값을 원본(unscaled) test_data에 추가
    test_data['rf_pred'] = y_pred_test_rf

    test_trip_groups = test_data.groupby('trip_id')
    hybrid_integrals_test, data_integrals_test = [], []
    processed_trips = 0
    integration_errors = 0

    for trip_id, group in test_trip_groups:
        if len(group) < 2: continue
        try:
            hybrid_integral, data_integral = integrate_and_compare(group.copy())
            if abs(data_integral) > 1e-6:
                 hybrid_integrals_test.append(hybrid_integral)
                 data_integrals_test.append(data_integral)
                 processed_trips += 1
        except Exception as e:
            print(f"Error during integration for trip {trip_id}: {e}")
            integration_errors += 1

    mape_test = calculate_mape(np.array(data_integrals_test), np.array(hybrid_integrals_test)) if processed_trips > 0 else float('nan')

    print(f"\nTest Set Integration Metrics ({processed_trips} trips processed, {integration_errors} errors):")
    print(f"MAPE (Energy): {mape_test:.2f}%" if not np.isnan(mape_test) else "MAPE: Not Available")

    results = [{
        'rmse': test_rmse,
        'test_mape': mape_test,
        'model_params': rf_model.get_params() # 사용된 파라미터 저장
    }]

    # -----------------------
    # 7. Plotting (선택적, RF에 맞는 플롯 추가 가능 - 예: Feature Importance)
    # -----------------------
    if plot:
        print("Generating feature importance plot...")
        try:
            import matplotlib.pyplot as plt
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feature_names = FEATURE_COLS

            plt.figure(figsize=(12, 6))
            plt.title(f"Feature Importances for Random Forest ({selected_car})")
            plt.bar(range(X_train.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
            plt.xlim([-1, X_train.shape[1]])
            plt.tight_layout()

            # 플랫폼에 따라 저장 경로 설정
            if platform.system() == "Windows":
                save_fig_dir = r"C:\Users\BSL\Desktop\Figures\RF_Importance"
            else:
                save_fig_dir = os.path.expanduser("~/SamsungSTF/Figures/RF_Importance")

            if not os.path.exists(save_fig_dir):
                os.makedirs(save_fig_dir)

            fig_save_path = os.path.join(save_fig_dir, f"RF_FeatureImportance_{selected_car}.png")
            plt.savefig(fig_save_path, dpi=300)
            print(f"Feature importance plot saved to {fig_save_path}")
            plt.show()

        except Exception as e:
            print(f"Could not generate feature importance plot: {e}")

    # -----------------------
    # 8. 모델 및 스케일러 저장
    # -----------------------
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save Random Forest Model using pickle
        model_file = os.path.join(save_dir, f"RF_model_{selected_car}.pkl")
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(rf_model, f)
            print(f"Random Forest model for {selected_car} saved to {model_file}")
        except Exception as e:
            print(f"Error saving Random Forest model: {e}")

        # Save Scaler
        scaler_path = os.path.join(save_dir, f'RF_scaler_{selected_car}.pkl')
        try:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved at {scaler_path}")
        except Exception as e:
            print(f"Error saving scaler: {e}")

    end_workflow_time = time.time()
    print(f"Random Forest workflow for {selected_car} completed in {end_workflow_time - start_workflow_time:.2f} seconds.")

    return results, scaler