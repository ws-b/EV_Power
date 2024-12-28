import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from GS_Functions import calculate_mape
from scipy.integrate import cumulative_trapezoid
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from optuna.trial import TrialState

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
# 파일 처리 함수
# ----------------------------
def process_single_file(file, trip_id):
    """
    파일 하나를 읽어 해당 파일이 하나의 trip으로 가정하고,
    rolling feature를 계산한 뒤 데이터를 반환.
    """
    try:
        data = pd.read_csv(file)

        # Power_data, Power_phys 컬럼이 모두 있는 경우만 처리
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

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
    """
    if scaler is None:
        min_vals = [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN]
        max_vals = [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX]
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
    예측치(Power_ml)와 실제 계측치(Power_data)를 적분하여 비교
    """
    trip_data = trip_data.sort_values(by='time')
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # Power_ml = 모델이 예측한 값 (직접 Power_data를 예측하므로 y_pred)
    trip_data['Power_ml'] = trip_data['y_pred']

    ml_cum_integral = cumulative_trapezoid(trip_data['Power_ml'].values, time_seconds, initial=0)
    ml_integral = ml_cum_integral[-1]

    data_cum_integral = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)
    data_integral = data_cum_integral[-1]

    return ml_integral, data_integral


# ----------------------------
# 파일기반 K-Fold용 CV Objective
# ----------------------------
def cv_objective(trial, train_files, scaler):
    # 하이퍼파라미터 샘플링
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

        fold_train_data_list = []
        for i, f in enumerate(fold_train_files):
            d = process_single_file(f, trip_id=i)
            if d is not None:
                fold_train_data_list.append(d)
        fold_train_data = pd.concat(fold_train_data_list, ignore_index=True)

        fold_val_data_list = []
        for j, f in enumerate(fold_val_files):
            d = process_single_file(f, trip_id=1000 + j)
            if d is not None:
                fold_val_data_list.append(d)
        fold_val_data = pd.concat(fold_val_data_list, ignore_index=True)

        # 스케일링
        fold_train_data_scaled, _ = scale_data(fold_train_data.copy(), scaler)
        fold_val_data_scaled, _ = scale_data(fold_val_data.copy(), scaler)

        # -----------------------------
        # **중요 변경**: label = Power_data
        # -----------------------------
        X_tr = fold_train_data_scaled[FEATURE_COLS]
        y_tr = fold_train_data_scaled['Power_data']  # ← Residual 대신 Power_data 사용
        X_val = fold_val_data_scaled[FEATURE_COLS]
        y_val = fold_val_data_scaled['Power_data']  # ← 동일

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
    # Optuna 스터디 생성
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: cv_objective(trial, train_files, scaler), n_trials=100)

    print(f"Best trial: {study.best_trial.params}")
    if plot:
        # 모든 trial의 결과를 DataFrame으로 변환
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

        # 최적 trial 정보 추출
        best_trial = study.best_trial
        best_trial_number = best_trial.number
        best_trial_value = best_trial.value

        # 최적 trial을 빨간색으로 강조
        plt.plot(best_trial_number, best_trial_value, marker='o', markersize=12, color='red', label='Best Trial')

        # 축 및 제목 설정
        plt.xlabel('Trial')
        plt.ylabel('CVE (Mean of Fold RMSE)')
        plt.title('CVE per Trial during Bayesian Optimization')

        # 범례 추가
        plt.legend()

        # 최적 파라미터 표시
        best_params_str = '\n'.join([f"{k}: {v:.4f}" for k, v in best_trial.params.items()])
        plt.text(0.95, 0.95, best_params_str, transform=plt.gca().transAxes,
                 fontsize=10, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # 레이아웃 조정
        plt.tight_layout()

        # 그래프 저장 경로 설정
        save_directory = r"C:\Users\BSL\Desktop\Figures\Supplementary"
        save_filename = f"FigureS10_{selected_car}_only_ml.png"
        save_path = os.path.join(save_directory, save_filename)

        try:
            # 디렉토리가 존재하지 않으면 생성
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
                print(f"디렉토리가 생성되었습니다: {save_directory}")

            # 그래프 저장
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 저장되었습니다: {save_path}")
        except Exception as e:
            print(f"그래프 저장 중 오류가 발생했습니다: {e}")

        # 그래프 표시
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
def run_workflow(vehicle_files, selected_car, plot=False, save_dir=False, predefined_best_params=None):
    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]

    # 파일 단위 Train/Test 분할 (8:2)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

    # Train 데이터 처리
    train_data_list = []
    for i, f in enumerate(train_files):
        d = process_single_file(f, trip_id=i)
        if d is not None:
            train_data_list.append(d)
    train_data = pd.concat(train_data_list, ignore_index=True)

    # Scaler Fit
    train_data_scaled, scaler = scale_data(train_data.copy(), scaler=None)

    # Test 데이터 처리
    test_data_list = []
    for j, f in enumerate(test_files):
        d = process_single_file(f, trip_id=1000 + j)
        if d is not None:
            test_data_list.append(d)
    test_data = pd.concat(test_data_list, ignore_index=True)

    # Test 데이터 스케일링
    test_data_scaled, _ = scale_data(test_data.copy(), scaler)

    # -----------------------------
    # abel = Power_data
    # -----------------------------
    X_train = train_data_scaled[FEATURE_COLS]
    y_train = train_data_scaled['Power_data']
    X_test = test_data_scaled[FEATURE_COLS]
    y_test = test_data_scaled['Power_data']

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

    y_pred_test = bst.predict(xgb.DMatrix(X_test))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"Test RMSE with best_params: {test_rmse:.4f}")

    test_data_scaled['y_pred'] = y_pred_test

    # trip_id별 적분 테스트
    test_trip_groups = test_data_scaled.groupby('trip_id')
    ml_integrals_test, data_integrals_test = [], []
    for _, group in test_trip_groups:
        ml_integral, data_integral = integrate_and_compare(group)
        ml_integrals_test.append(ml_integral)
        data_integrals_test.append(data_integral)

    mape_test = calculate_mape(np.array(data_integrals_test), np.array(ml_integrals_test))

    print(f"Test Set Integration Metrics:")
    print(f"MAPE: {mape_test:.2f}%")

    results = []
    results.append({
        'rmse': test_rmse,
        'test_mape': mape_test,
        'best_params': best_params
    })

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
