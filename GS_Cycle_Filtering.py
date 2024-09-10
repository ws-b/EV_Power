import os
import pandas as pd
import shutil
from tqdm import tqdm

# 파일을 읽을 폴더 경로 설정
folder_path = r'D:\SamsungSTF\Processed_Data\TripByTrip'
city_cycle_folder = os.path.join(folder_path, 'City_Cycle')
highway_cycle_folder = os.path.join(folder_path, 'Highway_Cycle')

# 새로운 폴더 생성
os.makedirs(city_cycle_folder, exist_ok=True)
os.makedirs(highway_cycle_folder, exist_ok=True)

# 조건에 맞는 파일을 저장할 리스트
city_cycle_files = []
highway_cycle_files = []

# 폴더 내의 모든 CSV 파일 읽기
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file_name in tqdm(all_files):
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)

    # 필요한 컬럼이 있는지 확인
    if 'time' in df.columns and 'speed' in df.columns and 'acceleration' in df.columns:
        # 시간 차 계산 (마지막과 첫 번째 시간 차이)
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

        total_time = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()

        # City Cycle 필터 조건
        speed_mean = df['speed'].mean()
        max_speed = df['speed'].max()
        stop_time_ratio = len(df[df['speed'] == 0]) / len(df)

        # 마지막 속도가 0이어야 한다는 조건 추가
        if 15/3.6 <= speed_mean <= 30/3.6 and 600 <= total_time <= 1200 and max_speed < 70/3.6 and stop_time_ratio >= 0.05 \
                and df['speed'].iloc[-1] == 0:
            city_cycle_files.append(file_name)
            shutil.copy(file_path, os.path.join(city_cycle_folder, file_name))

        # Highway Cycle 필터링 조건
        high_speed_time_ratio = len(df[df['speed'] >= 70/3.6]) / len(df)

        # 마지막 속도가 0이어야 한다는 조건 추가
        if 70/3.6 <= speed_mean <= 80/3.6 and high_speed_time_ratio >= 0.5 and 1200 <= total_time <= 1800 and df['speed'].iloc[-1] == 0:
            highway_cycle_files.append(file_name)
            shutil.copy(file_path, os.path.join(highway_cycle_folder, file_name))


# 결과 출력
print("City Cycle 파일들:", city_cycle_files)
print("Highway Cycle 파일들:", highway_cycle_files)
