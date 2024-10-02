import os
import pandas as pd
import numpy as np
from pyproj import Transformer
from tqdm import tqdm

# 좌표 변환 준비 (EPSG:5179에서 EPSG:4326로 변환)
transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

# 처리할 폴더 경로 지정
folder_path = r'D:\SamsungSTF\Processed_Data\KOTI'

# Haversine 공식을 사용하여 두 좌표 사이의 거리를 미터로 계산하는 함수
def haversine_np(lon1, lat1, lon2, lat2):
    R = 6371000  # 지구 반지름 (미터)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

# 폴더 내의 모든 CSV 파일 읽기
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file_name in tqdm(all_files, desc="Processing CSV files"):
    file_path = os.path.join(folder_path, file_name)

    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 'time' 컬럼을 datetime 형식으로 변환
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

        # 시간 기준으로 정렬 (필요 시)
        df = df.sort_values('time').reset_index(drop=True)

        # 좌표를 EPSG:4326으로 변환 (EPSG:5179 -> EPSG:4326)
        longitudes, latitudes = transformer.transform(df['x'].values, df['y'].values)
        df['longitude'] = longitudes
        df['latitude'] = latitudes

        # 이전 행의 좌표를 가져오기 위해 shift
        df['prev_longitude'] = df['longitude'].shift(1)
        df['prev_latitude'] = df['latitude'].shift(1)

        # Haversine 공식을 사용하여 거리 계산
        df['distance'] = haversine_np(
            df['prev_longitude'],
            df['prev_latitude'],
            df['longitude'],
            df['latitude']
        ).fillna(0)  # 첫 행의 거리는 0

        # 시간 차이 계산 (초 단위)
        df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)  # 첫 행의 시간 차이는 0

        # 속도 계산 (미터/초)
        df['speed'] = df['distance'] / df['time_diff'].replace(0, 1e-6)  # 0으로 나누는 것을 방지
        df.loc[df['time_diff'] == 0, 'speed'] = 0  # time_diff가 0인 경우 속도는 0

        # 가속도 계산 (미터/초²)
        df['acceleration'] = df['speed'].diff() / df['time_diff'].replace(0, 1e-6)
        df.loc[df['time_diff'] == 0, 'acceleration'] = 0  # time_diff가 0인 경우 가속도는 0
        df['acceleration'] = df['acceleration'].fillna(0)

        # 불필요한 컬럼 제거
        df.drop(['prev_longitude', 'prev_latitude'], axis=1, inplace=True)

        # 무한대 및 NaN 값 처리
        df['acceleration'] = df['acceleration'].replace([np.inf, -np.inf], 0).fillna(0)

        # 첫 행의 속도 및 가속도를 0으로 설정 (이미 처리되었지만 명시적으로 설정)
        df.at[0, 'speed'] = 0.0
        df.at[0, 'acceleration'] = 0.0

        # 저장할 컬럼 정의
        columns_tosave = ['time', 'x', 'y', 'longitude', 'latitude', 'speed', 'acceleration']

        # CSV 파일에 저장
        df.to_csv(file_path, columns=columns_tosave, index=False)

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
