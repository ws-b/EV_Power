import pandas as pd
import numpy as np
import glob
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# 데이터 처리 함수
# ----------------------------

def process_single_file(file):
    """
    단일 CSV 파일을 처리하여 필요한 열을 선택합니다.
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
    여러 CSV 파일을 병렬로 처리하고, 특징을 스케일링합니다.
    """
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6  # km/h를 m/s로 변환
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
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                data = future.result()
                if data is not None:
                    # 'time' 열을 datetime으로 변환
                    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

                    # 윈도우 크기 5로 롤링 통계량 계산
                    data['mean_accel_10'] = data['acceleration'].rolling(window=5, min_periods=1).mean()
                    data['std_accel_10'] = data['acceleration'].rolling(window=5, min_periods=1).std().fillna(0)
                    data['mean_speed_10'] = data['speed'].rolling(window=5, min_periods=1).mean()
                    data['std_speed_10'] = data['speed'].rolling(window=5, min_periods=1).std().fillna(0)

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

def train_model_linear_regression(X_train, y_train):
    """
    선형 회귀 모델을 훈련시킵니다.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ----------------------------
# 메인 코드
# ----------------------------

def main():
    # CSV 파일이 있는 폴더 경로 정의
    folder_path = r'D:\SamsungSTF\Processed_Data\TripByTrip'
    NiroEV = ['01241228149', '01241228151', '01241228153', '01241228154', '01241228155']
    Ioniq6 = ['01241248713', '01241592904', '01241597763', '01241597804']

    # 폴더 내 모든 CSV 파일 가져오기
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    filtered_files = [file for file in all_files if any(keyword in os.path.basename(file) for keyword in Ioniq6)]

    # 충분한 파일이 있는지 확인
    if len(filtered_files) < 51:
        print("Not enough files to perform this operation.")
        return
    for i in range(10):
        # 학습 및 테스트 파일 무작위 선택
        np.random.seed(i+42)
        all_indices = np.arange(len(filtered_files))
        np.random.shuffle(all_indices)

        # 학습용 파일 선택 (샘플 수 10개)
        train_indices_10 = all_indices[:10]
        train_files_10 = [filtered_files[i] for i in train_indices_10]

        # 학습용 파일 선택 (샘플 수 50개)
        train_indices_50 = all_indices[:50]
        train_files_50 = [filtered_files[i] for i in train_indices_50]

        # 학습에 사용되지 않은 테스트 파일 선택
        test_indices = all_indices[51:]  # 학습에 사용된 51개 파일 제외
        if not len(test_indices):
            print("Not enough files for testing.")
            return
        test_index = test_indices[0]
        test_file = filtered_files[test_index]

        # 샘플 수 10개로 학습 데이터 처리
        train_data_10, scaler_10 = process_files(train_files_10)
        feature_cols = [
            'speed', 'acceleration', 'ext_temp',
            'mean_accel_10', 'std_accel_10',
            'mean_speed_10', 'std_speed_10'
        ]
        X_train_10 = train_data_10[feature_cols]
        y_train_10 = train_data_10['Power_data']

        # 모델 학습 (샘플 수 10개)
        model_10 = train_model_linear_regression(X_train_10, y_train_10)

        # 샘플 수 50개로 학습 데이터 처리
        train_data_50, scaler_50 = process_files(train_files_50)
        X_train_50 = train_data_50[feature_cols]
        y_train_50 = train_data_50['Power_data']

        # 모델 학습 (샘플 수 50개)
        model_50 = train_model_linear_regression(X_train_50, y_train_50)

        # 테스트 파일 처리
        test_data_raw = process_single_file(test_file)
        if test_data_raw is None:
            print(f"Failed to process test file: {test_file}")
            return

        # 'time' 열을 datetime으로 변환
        test_data_raw['time'] = pd.to_datetime(test_data_raw['time'], format='%Y-%m-%d %H:%M:%S')

        # 롤링 통계량 계산
        test_data_raw['mean_accel_10'] = test_data_raw['acceleration'].rolling(window=5, min_periods=1).mean()
        test_data_raw['std_accel_10'] = test_data_raw['acceleration'].rolling(window=5, min_periods=1).std().fillna(0)
        test_data_raw['mean_speed_10'] = test_data_raw['speed'].rolling(window=5, min_periods=1).mean()
        test_data_raw['std_speed_10'] = test_data_raw['speed'].rolling(window=5, min_periods=1).std().fillna(0)

        # 학습 데이터의 스케일러를 사용하여 스케일링 적용
        test_data_10 = test_data_raw.copy()
        test_data_10[feature_cols] = scaler_10.transform(test_data_raw[feature_cols])

        test_data_50 = test_data_raw.copy()
        test_data_50[feature_cols] = scaler_50.transform(test_data_raw[feature_cols])

        # 테스트 데이터 준비
        X_test_10 = test_data_10[feature_cols]
        X_test_50 = test_data_50[feature_cols]

        # 모델을 사용하여 예측 수행
        test_data_10['Power_hybrid'] = model_10.predict(X_test_10) / 1000
        test_data_50['Power_hybrid'] = model_50.predict(X_test_50) / 1000
        test_data_10['Power_data'] = test_data_10['Power_data'] / 1000
        test_data_50['Power_data'] = test_data_50['Power_data'] / 1000

        # 경과 시간을 계산하여 추가
        test_data_10['elapsed_time'] = (test_data_10['time'] - test_data_10['time'].iloc[0]).dt.total_seconds()
        test_data_50['elapsed_time'] = (test_data_50['time'] - test_data_50['time'].iloc[0]).dt.total_seconds()

        # 누적 차이 계산 (Power_data - Power_hybrid)
        test_data_10['error_10'] = test_data_10['Power_data'] - test_data_10['Power_hybrid']
        test_data_50['error_50'] = test_data_50['Power_data'] - test_data_50['Power_hybrid']

        test_data_10['cumul_error_10'] = np.cumsum(test_data_10['Power_data'] - test_data_10['Power_hybrid'])
        test_data_50['cumul_error_50'] = np.cumsum(test_data_50['Power_data'] - test_data_50['Power_hybrid'])

        # 결과를 플롯팅
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # 첫 번째 플롯: Power_data와 Power_hybrid 비교
        axs[0].plot(test_data_10['elapsed_time'], test_data_10['Power_data'], label='Power_data', color='blue')
        axs[0].plot(test_data_10['elapsed_time'], test_data_10['Power_hybrid'], label='Power_hybrid (Model trained on 10 samples)', color='red', linestyle='--', alpha=0.7)
        axs[0].plot(test_data_50['elapsed_time'], test_data_50['Power_hybrid'], label='Power_hybrid (Model trained on 50 samples)', color='green', linestyle='--', alpha=0.7)
        axs[0].set_ylabel('Power (kW)')
        axs[0].set_title('Comparison of Power_data and Power_hybrid')
        axs[0].legend(loc='upper left')

        # 두 번째 플롯: 차이
        axs[1].plot(test_data_10['elapsed_time'], test_data_10['error_10'], label='Error (10 samples)', color='red', linestyle='--', alpha=0.7)
        axs[1].plot(test_data_50['elapsed_time'], test_data_50['error_50'], label='Error (50 samples)', color='green', linestyle='--', alpha=0.7)
        axs[1].set_xlabel('Elapsed Time (s)')
        axs[1].set_ylabel('Error (kW)')
        axs[1].set_title('Difference between Power_data and Power_hybrid')
        axs[1].legend(loc='upper left')

        # 세 번째 플롯: 누적 차이
        axs[2].plot(test_data_10['elapsed_time'], test_data_10['cumul_error_10'], label='Error (10 samples)', color='red', linestyle='--', alpha=0.7)
        axs[2].plot(test_data_50['elapsed_time'], test_data_50['cumul_error_50'], label='Error (50 samples)', color='green', linestyle='--', alpha=0.7)
        axs[2].set_xlabel('Elapsed Time (s)')
        axs[2].set_ylabel('Cumulative Error (kW)')
        axs[2].set_title('Cumulative Difference between Power_data and Power_hybrid')
        axs[2].legend(loc='upper left')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
