import os
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
# 모델과 스케일러 로드
def load_model_and_scaler(model_path, scaler_path):
    """
    XGBoost 모델과 스케일러를 로드합니다.

    Parameters:
        model_path (str): 저장된 XGBoost 모델 파일 경로.
        scaler_path (str): 저장된 스케일러 파일 경로.

    Returns:
        model (xgb.Booster): 로드된 XGBoost 모델.
        scaler (MinMaxScaler): 로드된 스케일러.
    """
    # XGBoost 모델 로드
    model = xgb.Booster()
    model.load_model(model_path)
    print(f"모델이 {model_path} 에서 로드되었습니다.")

    # 스케일러 로드
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"스케일러가 {scaler_path} 에서 로드되었습니다.")

    return model, scaler


def process_single_new_file(file_path, model, scaler):
    """
    단일 CSV 파일을 처리하여 'Power_hybrid' 컬럼을 추가하고, 원본 파일을 덮어씁니다.

    Parameters:
        file_path (str): 처리할 CSV 파일의 경로.
        model (xgb.Booster): 로드된 XGBoost 모델.
        scaler (MinMaxScaler): 로드된 스케일러.

    Returns:
        None
    """
    try:
        # CSV 파일 읽기
        data = pd.read_csv(file_path)
        print(f"파일 처리 중: {file_path}")

        # 필수 컬럼 확인
        required_cols = ['time', 'speed', 'acceleration', 'ext_temp', 'Power_phys']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"{file_path} 에 누락된 컬럼: {missing_cols}")

        # 'time' 컬럼을 datetime으로 변환
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

        # 롤링 통계량 계산 (윈도우 크기 = 5)
        data['mean_accel_10'] = data['acceleration'].rolling(window=5, min_periods=1).mean()
        data['std_accel_10'] = data['acceleration'].rolling(window=5).std().fillna(0)
        data['mean_speed_10'] = data['speed'].rolling(window=5).mean()
        data['std_speed_10'] = data['speed'].rolling(window=5).std().fillna(0)

        # 특성 컬럼 정의 (훈련 시 사용한 것과 동일해야 함)
        feature_cols = [
            'speed', 'acceleration', 'ext_temp',
            'mean_accel_10', 'std_accel_10',
            'mean_speed_10', 'std_speed_10'
        ]

        # 모든 특성 컬럼이 있는지 확인
        missing_features = [col for col in feature_cols if col not in data.columns]
        if missing_features:
            raise ValueError(f"{file_path} 처리 후 누락된 특성 컬럼: {missing_features}")

        # 특성 스케일링
        scaled_features = scaler.transform(data[feature_cols])

        # DMatrix 생성
        dmatrix = xgb.DMatrix(scaled_features, feature_names=feature_cols)

        # 예측 수행
        y_pred = model.predict(dmatrix)
        data['y_pred'] = y_pred

        # 'Power_hybrid' 계산
        data['Power_hybrid'] = data['Power_phys'] + data['y_pred']

        # 불필요한 컬럼 제거
        columns_to_drop = ['mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10', 'y_pred']
        data.drop(columns=columns_to_drop, inplace=True)
        print(f"불필요한 컬럼이 제거되었습니다: {columns_to_drop}")

        # 'Power_hybrid' 컬럼이 추가된 데이터프레임을 원본 파일에 저장
        data.to_csv(file_path, index=False)
        print(f"'Power_hybrid' 컬럼이 {file_path} 에 추가되고 파일이 덮어쓰기 되었습니다.")

    except Exception as e:
        print(f"{file_path} 처리 중 오류 발생: {e}")


def process_multiple_new_files(file_paths, model, scaler):
    """
    여러 CSV 파일을 병렬로 처리하여 'Power_hybrid' 컬럼을 추가하고 원본 파일을 덮어씁니다.

    Parameters:
        file_paths (list): 처리할 CSV 파일들의 경로 리스트.
        model (xgb.Booster): 로드된 XGBoost 모델.
        scaler (MinMaxScaler): 로드된 스케일러.

    Returns:
        None
    """
    # ProcessPoolExecutor를 사용하여 병렬 처리
    with ProcessPoolExecutor() as executor:
        # 각 파일에 대해 process_single_new_file 함수를 실행
        futures = [
            executor.submit(process_single_new_file, file_path, model, scaler)
            for file_path in file_paths
        ]

        # 모든 작업이 완료될 때까지 기다림
        for future in as_completed(futures):
            try:
                future.result()  # 예외가 발생하면 여기서 처리됨
            except Exception as e:
                print(f"파일 처리 중 예외 발생: {e}")
def main():
    # 모델과 스케일러 파일 경로 설정
    model_path = "D:\SamsungSTF\Processed_Data\Models\XGB_best_model_EV6.model"
    scaler_path = "D:\SamsungSTF\Processed_Data\Models\XGB_scaler_EV6.pkl"

    # CSV 파일들이 위치한 디렉토리 설정
    csv_folder = r"D:\SamsungSTF\Data\Cycle\HW_KOTI" # 실제 CSV 파일들이 있는 폴더 경로로 변경

    # 처리할 CSV 파일 리스트 가져오기
    file_paths = [os.path.join(csv_folder, file) for file in os.listdir(csv_folder) if file.endswith('.csv')]

    if not file_paths:
        print(f"지정된 폴더에 CSV 파일이 없습니다: {csv_folder}")
        return

    # 모델과 스케일러 로드
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    # 여러 파일을 병렬로 처리
    process_multiple_new_files(file_paths, model, scaler)


if __name__ == "__main__":
    main()
