import os
import glob
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scipy.spatial import cKDTree
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# 1. 데이터 로드 및 전처리를 위한 함수 정의

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    return df


def load_stations(stations_file):
    stations = pd.read_csv(stations_file)
    stations['longitude'] = stations['LON'].astype(float)
    stations['latitude'] = stations['LAT'].astype(float)
    return stations


def find_nearest_stations(df, stations):
    station_coords = stations[['longitude', 'latitude']].values
    tree = cKDTree(station_coords)
    data_coords = df[['longitude', 'latitude']].values
    distances, indices = tree.query(data_coords, k=1)
    df['STN_ID'] = stations.iloc[indices]['STN_ID'].values
    return df


def prepare_unique_requests(df):
    df['tm'] = df['time'].dt.strftime('%Y%m%d%H00')  # 'yyyymmddHH00' 형식
    unique_requests = df[['tm', 'STN_ID']].drop_duplicates()
    return unique_requests


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


def fetch_observation_data(unique_requests, auth_key, help_param=0, max_workers=10):
    observation_data = {}

    def fetch_and_store_text(row):
        tm = row['tm']
        stn = row['STN_ID']
        data = get_observation_data_text(tm, stn, auth_key, help_param)
        if data:
            observation_data[(tm, stn)] = data
        # API 호출 간 간단한 지연을 추가하여 서버 부하를 줄일 수 있습니다.
        time.sleep(0.1)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_and_store_text, row) for idx, row in unique_requests.iterrows()]
        for future in as_completed(futures):
            pass  # 모든 작업이 완료될 때까지 대기

    return observation_data


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


def parse_all_observations(observation_data):
    parsed_data = {}
    for key, text_content in observation_data.items():
        data = parse_observation_text(text_content)
        parsed_data[key] = data
    return parsed_data


def add_external_temp(df, parsed_data):
    def extract_TA_text(row):
        key = (row['tm'], row['STN_ID'])
        data = parsed_data.get(key, {})
        try:
            return data.get('TA', np.nan)  # 'TA' 필드가 없으면 NaN 반환
        except (ValueError, TypeError):
            return np.nan

    df['ext_temp'] = df.apply(extract_TA_text, axis=1)
    return df


def save_processed_data(df, original_file):
    # 저장할 컬럼 정의
    columns_tosave = ['time', 'x', 'y', 'longitude', 'latitude', 'speed', 'acceleration', 'ext_temp']

    # 원본 파일에 덮어쓰기
    df.to_csv(original_file, columns=columns_tosave, index=False)
    print(f"데이터 저장 완료: {original_file}")


# 2. 파일별 처리 함수 정의

def process_file(file_path, stations, auth_key, help_param=0):
    print(f"Processing file: {file_path}")

    # 데이터 로드
    df = load_data(file_path)

    # 가장 가까운 지상관측소 찾기
    df = find_nearest_stations(df, stations)

    # 고유한 API 요청 준비
    unique_requests = prepare_unique_requests(df)

    # API 호출을 통해 관측 데이터 가져오기
    observation_data = fetch_observation_data(unique_requests, auth_key, help_param)

    # 모든 API 응답 파싱
    parsed_data = parse_all_observations(observation_data)

    # 외부 온도 데이터 추가
    df = add_external_temp(df, parsed_data)

    # 원본 파일에 덮어쓰기
    save_processed_data(df, file_path)


# 3. 메인 함수 정의

def main():
    # .env 파일 로드
    load_dotenv()
    auth_key = os.getenv('KMA_API_KEY')
    if not auth_key:
        print("Error: KMA_API_KEY not found in environment variables.")
        return

    help_param = 0

    # 폴더 경로 정의
    input_folder = r"D:\SamsungSTF\Data\Cycle\HW_KOTI"
    stations_file = r"D:\SamsungSTF\Data\KMA\Stations.csv"

    # 관측소 데이터 로드 (한 번만 로드)
    stations = load_stations(stations_file)

    # 입력 폴더 내 모든 CSV 파일 목록 가져오기
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if not csv_files:
        print(f"폴더 내에 CSV 파일이 없습니다: {input_folder}")
        return

    print(f"처리할 CSV 파일 수: {len(csv_files)}개")

    # 파일별로 순차 처리 (병렬 처리 시 API 호출 제한 주의)
    for file in csv_files:
        try:
            process_file(file, stations, auth_key, help_param)
        except Exception as e:
            print(f"파일 처리 중 에러 발생 ({file}): {e}")

    print("모든 파일이 처리되었습니다.")


if __name__ == "__main__":
    main()
