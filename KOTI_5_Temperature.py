import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from scipy.spatial import cKDTree
import requests
import time  # API 호출 간 시간 지연을 위해 추가
from concurrent.futures import ThreadPoolExecutor, as_completed  # 병렬 처리를 위해 추가

# 1. 데이터 불러오기
file = r"D:\SamsungSTF\Data\Cycle\HW_KOTI\20190420_903436.csv"
df = pd.read_csv(file)

# 'time' 컬럼을 datetime 형식으로 변환
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

# 지상관측소 데이터
stations = pd.read_csv(r"D:\SamsungSTF\Data\KMA\Stations.csv")

# 지상관측소의 경도와 위도가 숫자형인지 확인 및 변환
stations['longitude'] = stations['LON'].astype(float)
stations['latitude'] = stations['LAT'].astype(float)

# 2. KDTree를 이용한 최근접 지상관측소 찾기
# 지상관측소의 좌표 배열 생성
station_coords = stations[['longitude', 'latitude']].values
tree = cKDTree(station_coords)

# 메인 데이터의 좌표 배열 생성
data_coords = df[['longitude', 'latitude']].values

# 각 데이터 포인트에 대해 가장 가까운 지상관측소의 인덱스 찾기
distances, indices = tree.query(data_coords, k=1)

# 가장 가까운 지상관측소의 STN_ID를 메인 데이터에 추가
df['STN_ID'] = stations.iloc[indices]['STN_ID'].values

# 3. API 요청을 위한 시간 형식 변환
# 'tm' 값을 정각으로 설정 (분을 00으로) - floor 대신 strftime을 사용하여 직접 설정
df['tm'] = df['time'].dt.strftime('%Y%m%d%H00')  # 'yyyymmddHH00' 형식

# 4. API 요청을 위한 고유한 (tm, STN_ID) 조합 생성
unique_requests = df[['tm', 'STN_ID']].drop_duplicates()

# 5. API 요청 함수 정의 (텍스트 응답 처리)
def get_observation_data_text(tm, stn, auth_key, help_param=0):
    url = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php'
    params = {
        'tm': tm,
        'stn': stn,
        'help': help_param,
        'authKey': auth_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # HTTP 에러 발생 시 예외 발생
        return response.text  # 텍스트 형식으로 응답을 반환
    except requests.exceptions.RequestException as e:
        print(f"Request failed for tm={tm}, stn={stn}: {e}")
        return None

# 6. API 키 설정
# .env 파일 로드
load_dotenv()
auth_key = os.getenv('KMA_API_KEY')
help_param = 0

# 7. API 호출 및 데이터 수집
observation_data = {}

# 병렬 처리를 위한 함수 정의
def fetch_and_store_text(row):
    tm = row['tm']
    stn = row['STN_ID']
    data = get_observation_data_text(tm, stn, auth_key, help_param)
    if data:
        observation_data[(tm, stn)] = data
    # API 호출 간 간단한 지연을 추가하여 서버 부하를 줄일 수 있습니다.
    time.sleep(0.1)
    return

# 병렬로 API 호출 수행
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_and_store_text, row) for idx, row in unique_requests.iterrows()]
    for future in as_completed(futures):
        pass  # 모든 작업이 완료될 때까지 대기

# 8. API 응답 데이터 파싱 (텍스트 형식)
def parse_observation_text(text_content):
    try:
        lines = text_content.splitlines()
        data_line = None
        for line in lines:
            if line.startswith('#'):
                continue  # 주석 라인 건너뛰기
            if line.strip() == '':
                continue  # 빈 라인 건너뛰기
            data_line = line.strip()
            break  # 첫 번째 데이터 라인 찾기

        if not data_line:
            print("No data line found in the response.")
            return {'TA': np.nan}

        # 데이터 라인을 공백으로 분할
        fields = data_line.split()

        if len(fields) < 12:
            print("Insufficient number of fields in the data line.")
            return {'TA': np.nan}

        # 'TA'는 12번째 필드 (인덱스 11)
        ta = fields[11]

        # 'TA'가 숫자인지 확인하고, 숫자가 아니면 NaN으로 처리
        try:
            ta_float = float(ta)
        except ValueError:
            ta_float = np.nan

        return {'TA': ta_float}
    except Exception as e:
        print(f"Error parsing observation text: {e}")
        return {'TA': np.nan}

parsed_data = {}

for key, text_content in observation_data.items():
    data = parse_observation_text(text_content)
    parsed_data[key] = data

# 9. 파싱된 데이터를 메인 데이터프레임에 통합
# 'TA' 필드만 추출하여 'ext_temp' 컬럼에 추가
def extract_TA_text(row):
    key = (row['tm'], row['STN_ID'])
    data = parsed_data.get(key, {})
    try:
        return data.get('TA', np.nan)  # 'TA' 필드가 없으면 NaN 반환
    except (ValueError, TypeError):
        return np.nan

df['ext_temp'] = df.apply(extract_TA_text, axis=1)
columns_tosave = ['time', 'x', 'y', 'longitude', 'latitude', 'speed', 'acceleration', 'ext_temp']
# 10. 최종 데이터 저장
df.to_csv(file, columns=columns_tosave , index=False)
print(f"데이터 저장 완료: {file}")