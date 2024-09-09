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

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

# 폴더 내의 모든 CSV 파일 읽기
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file_name in tqdm(all_files):
    file_path = os.path.join(folder_path, file_name)

    # CSV 파일 읽기
    df = pd.read_csv(file_path)

    # 'time' 컬럼을 datetime 형식으로 변환
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

    # 좌표를 EPSG:4326로 변환 (EPSG:5179 -> EPSG:4326)
    longitudes, latitudes = transformer.transform(df['x'].values, df['y'].values)
    df['longitude'] = longitudes
    df['latitude'] = latitudes

    # 시간 차이 계산 (초 단위) - NumPy 배열 사용
    time_diff = np.diff(df['time'].values).astype('timedelta64[s]').astype(float)
    time_diff = np.insert(time_diff, 0, 0)  # 첫 번째 값은 0으로 설정

    # 거리 계산 (NumPy 배열로 Haversine 공식 적용)
    distances = haversine_np(
        df['longitude'][:-1], df['latitude'][:-1],
        df['longitude'][1:], df['latitude'][1:]
    )
    distances = np.insert(distances, 0, 0)  # 첫 번째 값은 0으로 설정

    # 속도 계산 (거리 / 시간차) -> 첫 번째 행은 계산 제외
    speeds = np.zeros_like(distances)
    speeds[1:] = distances[1:] / time_diff[1:]

    # 가속도 계산 (속도 변화량 / 시간차)
    accelerations = np.zeros_like(speeds)
    accelerations[2:] = np.diff(speeds[1:]) / time_diff[2:]

    # 속도를 km/h로 변환
    speeds *= 3.6

    # 새로운 컬럼들 추가
    df['time_diff'] = time_diff
    df['distance'] = distances
    df['speed'] = speeds
    df['acceleration'] = accelerations

    # 저장할 컬럼 정의
    columns_tosave = ['time', 'x', 'y', 'longitude', 'latitude', 'speed', 'acceleration']

    # CSV 파일에 저장
    df.to_csv(file_path, columns=columns_tosave, index=False)
