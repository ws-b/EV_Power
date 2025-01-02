import os
import pandas as pd
import numpy as np
from pyproj import Transformer
from tqdm import tqdm

# EPSG:5179(한국좌표) → EPSG:4326(WGS84)로 좌표를 변환하기 위해 Transformer 객체를 생성합니다.
# always_xy=True 옵션은 변환 시 (x, y) 순서를 유지하도록 설정합니다.
transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

# CSV 파일들이 저장되어 있는 폴더 경로를 지정합니다.
folder_path = r'D:\SamsungSTF\Processed_Data\KOTI'

# 두 좌표(경도, 위도) 사이의 거리를 계산하기 위해 Haversine 공식을 활용하는 함수입니다.
# 반환값은 두 점 사이의 거리(미터)입니다.
def haversine_np(lon1, lat1, lon2, lat2):
    R = 6371000  # 지구의 평균 반지름(단위: 미터)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (np.sin(delta_phi / 2.0) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

# 지정된 폴더 안에서 확장자가 .csv 인 모든 파일을 리스트로 가져옵니다.
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# tqdm을 이용해 CSV 파일을 하나씩 처리합니다.
for file_name in tqdm(all_files, desc="Processing CSV files"):
    file_path = os.path.join(folder_path, file_name)

    try:
        # CSV 파일을 pandas DataFrame 형태로 불러옵니다.
        df = pd.read_csv(file_path)

        # 'time' 컬럼을 datetime 형식으로 변환합니다.
        # '%Y-%m-%d %H:%M:%S' 형식에 맞춰 parsing 합니다.
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

        # time 컬럼을 기준으로 정렬합니다(필요 시). 이후 인덱스를 재설정합니다.
        df = df.sort_values('time').reset_index(drop=True)

        # EPSG:5179 좌표(x, y)를 EPSG:4326(경도, 위도)로 변환합니다.
        longitudes, latitudes = transformer.transform(df['x'].values, df['y'].values)
        df['longitude'] = longitudes
        df['latitude'] = latitudes

        # 이전 행의 경도, 위도를 가져오기 위해 shift() 함수를 사용합니다.
        df['prev_longitude'] = df['longitude'].shift(1)
        df['prev_latitude'] = df['latitude'].shift(1)

        # Haversine 공식을 활용해 인접한 두 지점 간 거리(미터)를 계산합니다.
        df['distance'] = haversine_np(
            df['prev_longitude'],
            df['prev_latitude'],
            df['longitude'],
            df['latitude']
        )

        # 연속되는 두 시점 간 시간 차이를 초 단위로 계산합니다.
        df['time_diff'] = df['time'].diff().dt.total_seconds()

        # time_diff가 0이거나 NaN인 경우, 속도 및 가속도 계산에 오류가 생길 수 있으므로 해당 행을 제거합니다.
        df = df[df['time_diff'] != 0].reset_index(drop=True)

        # distance 또는 time_diff가 NaN인 행도 제거합니다.
        df = df.dropna(subset=['distance', 'time_diff']).reset_index(drop=True)

        # 속도를 계산합니다. (단위: m/s)
        df['speed'] = df['distance'] / df['time_diff']

        # 가속도를 계산합니다. (단위: m/s^2)
        # speed 간 차이를 time_diff로 나누어 계산합니다.
        df['acceleration'] = df['speed'].diff() / df['time_diff']

        # 첫 번째 행은 diff() 계산으로 가속도가 NaN이 되므로,
        # 임의로 두 번째 행의 가속도 값을 복사하여 채워줍니다. (가속도를 끊김 없이 연결하기 위함)
        df.at[0, 'acceleration'] = df.at[1, 'acceleration']

        # 중간 계산용으로 사용했던 불필요한 컬럼들은 제거합니다.
        df.drop(['prev_longitude', 'prev_latitude', 'distance', 'time_diff'], axis=1, inplace=True)

        # 최종적으로 변환 및 계산이 완료된 데이터를 다시 같은 파일 이름으로 저장합니다.
        df.to_csv(file_path, index=False)

    except Exception as e:
        # 파일 처리 중 오류가 발생할 경우 에러 메시지를 출력합니다.
        print(f"Error processing {file_name}: {e}")
