import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from GS_Functions import calculate_rrmse, calculate_rmse, calculate_mape
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from GS_plot import plot_contour, plot_shap_values
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import mean_squared_error
from scipy.integrate import cumulative_trapezoid

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
            return data[['time', 'speed', 'acceleration', 'ext_temp', 'Power_phys', 'Power_data']]
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
    트립 데이터에서 'y_pred'와 'Power_data'를 시간에 따라 적분합니다.

    Parameters:
        trip_data (pd.DataFrame): 특정 trip_id에 해당하는 데이터프레임

    Returns:
        tuple: (ml_integral, data_integral)
            ml_integral (float): y_pred의 총 에너지 (적분값)
            data_integral (float): Power_data의 총 에너지 (적분값)
    """
    # 'time'으로 정렬
    trip_data = trip_data.sort_values(by='time')

    # 'time'을 초 단위로 변환
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # 'y_pred'를 누적 트랩에즈 룰로 적분
    ml_cumulative = cumulative_trapezoid(trip_data['y_pred'].values, time_seconds, initial=0)
    ml_integral = ml_cumulative[-1]  # 총 적분값

    # 'Power_data'를 누적 트랩에즈 룰로 적분
    data_cumulative = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)
    data_integral = data_cumulative[-1]  # 총 적분값

    return ml_integral, data_integral

# ----------------------------
# Optuna Bayesian Optimization
# ----------------------------

def objective(trial, X_train, y_train, X_val, y_val):
    # 하이퍼파라미터 제안
    reg_lambda = trial.suggest_float('reg_lambda', 1e-6, 1e5, log=True)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-6, 1e5, log=True)
    eta = trial.suggest_float('eta', 0.01, 0.3, log=True)

    # DMatrix 생성
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 파라미터 설정
    params = {
        'tree_method': 'gpu_hist',
        'device': 'cuda:1',
        'eval_metric': 'rmse',
        'eta': eta,  # 학습률 설정
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'verbosity': 0,
        'objective': 'reg:squarederror'
    }

    # 조기 종료 설정
    evals = [(dval, 'validation')]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=15,
        verbose_eval=False
    )

    # 예측 및 RMSE 계산
    preds = bst.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    return rmse


def bayesian_optimization(trial_func, X_train, y_train, X_val, y_val, n_trials=50):
    """
    Optuna를 사용하여 하이퍼파라미터 튜닝을 수행합니다.
    이번 튜닝에서는 reg_lambda, reg_alpha, eta, num_boost_round를 최적화합니다.
    """
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: trial_func(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)

    print(f"Best trial: {study.best_trial.params}")
    return study.best_trial.params

# ----------------------------
# 모델 훈련 함수
# ----------------------------

def train_model(X_train, y_train, X_val, y_val, best_params):
    """
    최적의 하이퍼파라미터로 XGBoost 모델을 훈련시킵니다.
    """
    # DMatrix 생성
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 파라미터 설정
    params = {
        'tree_method': 'gpu_hist',
        'device': 'cuda:1',
        'eval_metric': 'rmse',
        'eta': best_params['eta'],
        'reg_lambda': best_params['reg_lambda'],
        'reg_alpha': best_params['reg_alpha'],
        'verbosity': 0,
        'objective': 'reg:squarederror'
    }

    # 모델 훈련
    evals = [(dval, 'validation')]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=15,
        verbose_eval=False
    )

    return bst

# ----------------------------
# 교차 검증 및 모델 훈련
# ----------------------------

def cross_validate(vehicle_files, selected_car, params=None, plot=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []
    best_params_overall = None

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]

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
        y_train = train_data['Power_data']

        X_test = test_data[feature_cols]
        y_test = test_data['Power_data']

        # 훈련 데이터를 추가로 훈련/검증 세트로 분할
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        if params is None:
            # Bayesian Optimization을 통해 하이퍼파라미터 튜닝
            best_params = bayesian_optimization(objective, X_tr, y_tr, X_val, y_val, n_trials=50)
            best_lambda = best_params['reg_lambda']
            best_alpha = best_params['reg_alpha']
        else:
            best_params = params
            best_lambda = best_params['reg_lambda']
            best_alpha = best_params['reg_alpha']

        # 모델 훈련
        bst = train_model(X_train, y_train, X_val, y_val, best_params)

        # 예측 수행
        train_data['y_pred'] = bst.predict(xgb.DMatrix(X_train))
        test_data['y_pred'] = bst.predict(xgb.DMatrix(X_test))

        if plot:
            # SHAP 값 계산 및 시각화
            plot_shap_values(bst, X_train, feature_cols, selected_car, None)

        # 트립별로 적분 수행
        train_trip_groups = train_data.groupby('trip_id')
        test_trip_groups = test_data.groupby('trip_id')

        ml_integrals_train, data_integrals_train = [], []
        for _, group in train_trip_groups:
            ml_integral, data_integral = integrate_and_compare(group)
            ml_integrals_train.append(ml_integral)
            data_integrals_train.append(data_integral)

        # 훈련 데이터의 MAPE 및 RRMSE 계산
        mape_train = calculate_mape(np.array(data_integrals_train), np.array(ml_integrals_train))
        rrmse_train = calculate_rrmse(np.array(data_integrals_train), np.array(ml_integrals_train))

        # 테스트 데이터의 적분 수행
        ml_integrals_test, data_integrals_test = [], []
        for _, group in test_trip_groups:
            ml_integral, data_integral = integrate_and_compare(group)
            ml_integrals_test.append(ml_integral)
            data_integrals_test.append(data_integral)

        # 테스트 데이터의 MAPE 및 RRMSE 계산
        mape_test = calculate_mape(np.array(data_integrals_test), np.array(ml_integrals_test))
        rrmse_test = calculate_rrmse(np.array(data_integrals_test), np.array(ml_integrals_test))

        # RMSE 계산
        rmse = calculate_rmse(
            y_test,
            test_data['y_pred']
        )

        # 결과 저장
        results.append({
            'fold': fold_num,
            'rmse': rmse,
            'test_rrmse': rrmse_test,
            'test_mape': mape_test,
            'best_params': best_params
        })
        models.append(bst)

        # 폴드 결과 출력
        print(f"--- Fold {fold_num} Results ---")
        print(
            f"Best Params: reg_lambda={best_lambda:.5f}, reg_alpha={best_alpha:.5f}")
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