import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import mean_squared_error
from optuna.trial import TrialState

# 사용자 정의 함수 (예: GS_Functions, GS_plot 등) 임포트
from GS_Functions import calculate_mape
from GS_plot import plot_shap_values, plot_composite_contour

# ----------------------------
# 전역 변수 / 상수 정의
# ----------------------------
SPEED_MIN = 0 / 3.6
SPEED_MAX = 230 / 3.6
ACCELERATION_MIN = -15
ACCELERATION_MAX = 9
TEMP_MIN = -30
TEMP_MAX = 50
ACCEL_STD_MAX = 10
SPEED_STD_MAX = 30

# slope 범위 (라디안 기준) - 실제 데이터 범위에 맞게 조정 가능
SLOPE_MIN = -1.57  # -90 degrees
SLOPE_MAX = 1.57  # +90 degrees

window_sizes = [5]


def generate_feature_columns():
    """
    모델에서 사용할 feature 컬럼 정의.
    기존 speed, acceleration, ext_temp에 slope를 추가하고,
    rolling mean/std도 기존처럼 포함합니다.
    """
    feature_cols = ['speed', 'acceleration', 'ext_temp', 'slope']
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
# 파일 처리 함수
# ----------------------------
def process_single_file(file, trip_id):
    """
    파일 하나를 읽어 해당 파일이 하나의 trip으로 가정하고,
    rolling feature와 slope(있으면) 계산 뒤 데이터를 반환.
    """
    try:
        data = pd.read_csv(file)

        # Power_phys & Power_data 모두 있어야 Residual 계산 가능
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

            # altitude가 있는 경우 slope 계산
            if 'altitude' in data.columns:
                # 속도, 시간 정보를 바탕으로 경사도(slope) 계산
                altitude_diff = np.diff(data['altitude'], prepend=data['altitude'].iloc[0])
                time_diff = data['time'].diff().dt.total_seconds().fillna(1).to_numpy()
                v = data['speed'].to_numpy()  # m/s 가정
                distance_diff = v * time_diff

                # np.where(distance_diff==0, 0, arctan2(...)) 처리
                slope_array = np.where(distance_diff == 0,
                                       0,
                                       np.arctan2(altitude_diff, distance_diff))
                data['slope'] = slope_array
            else:
                # slope가 없으면 NaN으로 채워줌
                data['slope'] = np.nan

            # rolling feature 계산
            for w in window_sizes:
                time_window = w * 2
                data[f'mean_accel_{time_window}'] = data['acceleration'].rolling(window=w, min_periods=1).mean().bfill()
                data[f'std_accel_{time_window}'] = data['acceleration'].rolling(window=w,
                                                                                min_periods=1).std().bfill().fillna(0)
                data[f'mean_speed_{time_window}'] = data['speed'].rolling(window=w, min_periods=1).mean().bfill()
                data[f'std_speed_{time_window}'] = data['speed'].rolling(window=w, min_periods=1).std().bfill().fillna(
                    0)

            data['trip_id'] = trip_id
            return data
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None


def scale_data(df, scaler=None):
    """
    FEATURE_COLS에 대해 MinMaxScaling.
    slope 범위를 -1.57 ~ +1.57(약 -90도~+90도)로 가정.
    """
    if scaler is None:
        # 기본 범위 설정
        # speed, accel, ext_temp 범위
        min_vals = [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN, SLOPE_MIN]
        max_vals = [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX, SLOPE_MAX]

        # windowed features: [mean_accel, std_accel, mean_speed, std_speed]
        # mean_accel   : ACCELERATION_MIN ~ ACCELERATION_MAX
        # std_accel    : 0 ~ ACCEL_STD_MAX
        # mean_speed   : SPEED_MIN ~ SPEED_MAX
        # std_speed    : 0 ~ SPEED_STD_MAX
        window_val_min = [ACCELERATION_MIN, 0, SPEED_MIN, 0]
        window_val_max = [ACCELERATION_MAX, ACCEL_STD_MAX, SPEED_MAX, SPEED_STD_MAX]

        for w in window_sizes:
            min_vals.extend(window_val_min)
            max_vals.extend(window_val_max)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame([min_vals, max_vals], columns=FEATURE_COLS))

    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    return df, scaler


def integrate_and_compare(trip_data):
    """
    time 축으로 적분하여 hybrid(= phys + y_pred) vs Power_data 비교 지표.
    """
    trip_data = trip_data.sort_values(by='time')
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # hybrid = 물리 모형 + ML이 예측한 잔차
    trip_data['Power_hybrid'] = trip_data['Power_phys'] + trip_data['y_pred']

    # 하이브리드/실측 각각 적분
    hybrid_cum_integral = cumulative_trapezoid(trip_data['Power_hybrid'].values, time_seconds, initial=0)
    data_cum_integral = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)

    return hybrid_cum_integral[-1], data_cum_integral[-1]


# ----------------------------
# 파일기반 K-Fold용 CV Objective
# ----------------------------
def cv_objective(trial, train_files, scaler):
    # Optuna 하이퍼파라미터 샘플링
    reg_lambda = trial.suggest_float('reg_lambda', 1e-6, 1e5, log=True)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-6, 1e5, log=True)
    eta = trial.suggest_float('eta', 0.01, 0.3, log=True)

    params = {
        'tree_method': 'gpu_hist',
        'device': 'cuda:0',
        'eval_metric': 'rmse',
        'eta': eta,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'verbosity': 0,
        'objective': 'reg:squarederror'
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []
    train_files = np.array(train_files)

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(train_files)):
        fold_train_files = train_files[train_idx]
        fold_val_files = train_files[val_idx]

        # Fold Train 데이터 생성
        fold_train_data_list = []
        for i, f in enumerate(fold_train_files):
            d = process_single_file(f, trip_id=i)
            if d is not None:
                fold_train_data_list.append(d)
        fold_train_data = pd.concat(fold_train_data_list, ignore_index=True)

        # Fold Val 데이터 생성
        fold_val_data_list = []
        for j, f in enumerate(fold_val_files):
            d = process_single_file(f, trip_id=1000 + j)
            if d is not None:
                fold_val_data_list.append(d)
        fold_val_data = pd.concat(fold_val_data_list, ignore_index=True)

        # 스케일링
        fold_train_data_scaled, _ = scale_data(fold_train_data.copy(), scaler)
        fold_val_data_scaled, _ = scale_data(fold_val_data.copy(), scaler)

        X_tr = fold_train_data_scaled[FEATURE_COLS]
        y_tr = fold_train_data_scaled['Residual']
        X_val = fold_val_data_scaled[FEATURE_COLS]
        y_val = fold_val_data_scaled['Residual']

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'validation')],
            early_stopping_rounds=15,
            verbose_eval=False
        )

        preds = bst.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_list.append(rmse)

    cve = np.mean(rmse_list)
    return cve


def tune_hyperparameters(train_files, scaler, selected_car, plot):
    # Optuna 스터디
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: cv_objective(trial, train_files, scaler), n_trials=100)

    print(f"Best trial: {study.best_trial.params}")
    if plot:
        # 모든 trial 결과 DataFrame 변환
        trials_df = study.trials_dataframe()

        # CSV 파일로 내보내기
        trials_save_path = r"C:\Users\BSL\Desktop\Results"
        trials_save = os.path.join(trials_save_path, f"{selected_car}_optuna_trials_results.csv")
        trials_df.to_csv(trials_save, index=False)
        print("All trial results have been saved to optuna_trials_results.csv")

        # 완료된 trials만 선택
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        trial_numbers = [t.number for t in complete_trials]
        trial_values = [t.value for t in complete_trials]

        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.plot(trial_numbers, trial_values, marker='o', linestyle='-', label='Trials')

        # 최적 trial 강조
        best_trial = study.best_trial
        best_trial_number = best_trial.number
        best_trial_value = best_trial.value
        plt.plot(best_trial_number, best_trial_value, marker='o', markersize=12, color='red', label='Best Trial')

        plt.xlabel('Trial')
        plt.ylabel('CVE (Mean of Fold RMSE)')
        plt.title('CVE per Trial during Bayesian Optimization')
        plt.legend()

        best_params_str = '\n'.join([f"{k}: {v:.4f}" for k, v in best_trial.params.items()])
        plt.text(0.95, 0.95, best_params_str, transform=plt.gca().transAxes,
                 fontsize=10, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.tight_layout()

        # 그래프 저장
        save_directory = r"C:\Users\BSL\Desktop\Figures\Supplementary"
        save_filename = f"FigureS10_{selected_car}.png"
        save_path = os.path.join(save_directory, save_filename)

        try:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
                print(f"디렉토리가 생성되었습니다: {save_directory}")

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 저장되었습니다: {save_path}")
        except Exception as e:
            print(f"그래프 저장 중 오류가 발생했습니다: {e}")

        plt.show()

    return study.best_trial.params


def train_final_model(X_train, y_train, best_params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        'tree_method': 'gpu_hist',
        'device': 'cuda:0',
        'eval_metric': 'rmse',
        'eta': best_params['eta'],
        'reg_lambda': best_params['reg_lambda'],
        'reg_alpha': best_params['reg_alpha'],
        'verbosity': 0,
        'objective': 'reg:squarederror'
    }
    bst = xgb.train(params, dtrain, num_boost_round=150, verbose_eval=False)
    return bst


# ----------------------------
# 워크플로우
# ----------------------------
def run_workflow(vehicle_files, selected_car, plot=False, save_dir="models", predefined_best_params=None):
    """
    vehicle_files: { selected_car: [파일경로1, 파일경로2, ...] } 형태의 dict
    selected_car: 'NiroEV', 'Ioniq5' 등 키
    """
    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]

    # Train/Test 분할
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

    # Train 데이터 로드
    train_data_list = []
    for i, f in enumerate(train_files):
        d = process_single_file(f, trip_id=i)
        if d is not None:
            train_data_list.append(d)
    train_data = pd.concat(train_data_list, ignore_index=True)

    # Scaler Fit
    train_data_scaled, scaler = scale_data(train_data.copy(), scaler=None)

    # Test 데이터 로드
    test_data_list = []
    for j, f in enumerate(test_files):
        d = process_single_file(f, trip_id=1000 + j)
        if d is not None:
            test_data_list.append(d)
    test_data = pd.concat(test_data_list, ignore_index=True)

    # Test 데이터 스케일링
    test_data_scaled, _ = scale_data(test_data.copy(), scaler)

    # X, y 분할
    X_train = train_data_scaled[FEATURE_COLS]
    y_train = train_data_scaled['Residual']

    X_test = test_data_scaled[FEATURE_COLS]
    y_test = test_data_scaled['Residual']

    # -----------------------
    # Hyperparameter Tuning or Skip
    # -----------------------
    if predefined_best_params is None:
        # Optuna로 하이퍼파라미터 튜닝
        best_params = tune_hyperparameters(train_files, scaler, selected_car, plot)
    else:
        # 이미 주어진 파라미터 사용
        best_params = predefined_best_params
        print(f"Using predefined best_params: {best_params}")

    # 최종 모델 학습
    bst = train_final_model(X_train, y_train, best_params)

    # Test 예측
    y_pred_test = bst.predict(xgb.DMatrix(X_test))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"Test RMSE with best_params: {test_rmse:.4f}")

    test_data_scaled['y_pred'] = y_pred_test

    # trip_id별로 적분
    test_trip_groups = test_data_scaled.groupby('trip_id')
    hybrid_integrals_test, data_integrals_test = [], []
    for _, group in test_trip_groups:
        hybrid_integral, data_integral = integrate_and_compare(group)
        hybrid_integrals_test.append(hybrid_integral)
        data_integrals_test.append(data_integral)

    mape_test = calculate_mape(np.array(data_integrals_test), np.array(hybrid_integrals_test))
    print(f"Test Set Integration Metrics:")
    print(f"MAPE: {mape_test:.2f}%")

    results = []
    results.append({
        'rmse': test_rmse,
        'test_mape': mape_test,
        'best_params': best_params
    })

    # 시각화 (선택)
    if plot:
        train_preds = bst.predict(xgb.DMatrix(X_train))
        train_data_scaled['y_pred'] = train_preds

        plot_composite_contour(
            X_train=X_train[['speed', 'acceleration']].values,
            y_pred_train=train_preds,
            X_test=X_test[['speed', 'acceleration']].values,
            y_pred_test1=y_pred_test,
            # 아래는 예시로 넣은 것으로, 필요 시 수정 가능
            y_pred_test2=(test_data_scaled['Residual'] - test_data_scaled['y_pred']).values,
            scaler=scaler,
            selected_car=selected_car,
            terminology1=f'{selected_car} : Train Residual',
            terminology2=f'{selected_car} : Residual[1]',
            terminology3=f'{selected_car} : Residual[2]',
            num_grids=30,
            min_count=10,
            save_directory=r"C:\Users\BSL\Desktop\Figures"
        )

        # SHAP 분석 예시 (원하시는 경우 주석 해제)
        # shap_value_save_path = fr"C:\Users\BSL\Desktop\Figures\Supplementary\FigureS11_{selected_car}.png"
        # plot_shap_values(bst, X_train, FEATURE_COLS, selected_car, shap_value_save_path)

    # 모델 및 스케일러 저장
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_file = os.path.join(save_dir, f"XGB_best_model_{selected_car}.model")
        bst.save_model(model_file)
        print(f"Best model for {selected_car} saved with Test RMSE: {test_rmse:.4f}")

        scaler_path = os.path.join(save_dir, f'XGB_scaler_{selected_car}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at {scaler_path}")

    return results, scaler


# ----------------------------
# 추가 기능 함수
# ----------------------------
def load_model_and_scaler(model_path, scaler_path):
    model = xgb.Booster()
    model.load_model(model_path)
    print(f"모델이 {model_path} 에서 로드되었습니다.")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"스케일러가 {scaler_path} 에서 로드되었습니다.")

    return model, scaler


def process_single_new_file(file_path, model, scaler):
    """
    학습된 모델+스케일러를 사용하여 새 CSV 파일에 'Power_hybrid' 컬럼을 추가하는 예시 함수.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"파일 처리 중: {file_path}")

        required_cols = ['time', 'speed', 'acceleration', 'ext_temp', 'Power_phys']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"{file_path} 에 누락된 컬럼: {missing_cols}")

        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

        # slope 계산 로직 (altitude 유무 확인)
        if 'altitude' in data.columns:
            altitude_diff = np.diff(data['altitude'], prepend=data['altitude'].iloc[0])
            time_diff = data['time'].diff().dt.total_seconds().fillna(1).to_numpy()
            v = data['speed'].to_numpy()
            distance_diff = v * time_diff
            slope_array = np.where(distance_diff == 0,
                                   0,
                                   np.arctan2(altitude_diff, distance_diff))
            data['slope'] = slope_array
        else:
            data['slope'] = np.nan

        # rolling feature
        for w in window_sizes:
            time_window = w * 2
            data[f'mean_accel_{time_window}'] = data['acceleration'].rolling(window=w, min_periods=1).mean().fillna(0)
            data[f'std_accel_{time_window}'] = data['acceleration'].rolling(window=w, min_periods=1).std().fillna(0)
            data[f'mean_speed_{time_window}'] = data['speed'].rolling(window=w, min_periods=1).mean().fillna(0)
            data[f'std_speed_{time_window}'] = data['speed'].rolling(window=w, min_periods=1).std().fillna(0)

        # 스케일링
        scaled_features = scaler.transform(data[FEATURE_COLS])
        dmatrix = xgb.DMatrix(scaled_features, feature_names=FEATURE_COLS)

        # 예측
        y_pred = model.predict(dmatrix)
        data['y_pred'] = y_pred
        data['Power_hybrid'] = data['Power_phys'] + data['y_pred']

        # 사용 뒤 불필요한 rolling 컬럼 등 제거 가능
        columns_to_drop = []
        for w in window_sizes:
            time_window = w * 2
            columns_to_drop.extend([
                f'mean_accel_{time_window}',
                f'std_accel_{time_window}',
                f'mean_speed_{time_window}',
                f'std_speed_{time_window}'
            ])
        # y_pred도 저장하지 않는다면 드롭할 수 있음
        columns_to_drop.append('y_pred')

        data.drop(columns=columns_to_drop, inplace=True)
        print(f"불필요한 컬럼이 제거되었습니다: {columns_to_drop}")

        data.to_csv(file_path, index=False)
        print(f"'Power_hybrid' 컬럼이 {file_path} 에 추가되고, 파일이 덮어쓰기 되었습니다.")
    except Exception as e:
        print(f"{file_path} 처리 중 오류 발생: {e}")


def process_multiple_new_files(file_paths, model, scaler):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_new_file, file_path, model, scaler)
            for file_path in file_paths
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"파일 처리 중 예외 발생: {e}")
