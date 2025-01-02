import os
import glob
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scipy.spatial import cKDTree
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor


# Custom exception for rate limit exceeded
# API 호출 시 Rate Limit(제한 횟수)를 초과했을 경우 발생시키는 예외입니다.
class RateLimitExceededError(Exception):
    """Exception raised when the API rate limit is exceeded."""
    pass


# ----------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리를 위한 함수 정의
# ----------------------------------------------------------------------------

def load_data(file_path):
    """
    지정된 CSV 파일을 로드해 'time' 컬럼을 datetime 형식으로 변환한 후 DataFrame을 반환합니다.
    """
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    return df


def load_stations(stations_file):
    """
    기상청 측정소 파일(Stations.csv)을 로드하여,
    LON과 LAT 컬럼을 float 형으로 변환한 후 DataFrame을 반환합니다.
    """
    stations = pd.read_csv(stations_file)
    stations['longitude'] = stations['LON'].astype(float)
    stations['latitude'] = stations['LAT'].astype(float)
    return stations


def find_nearest_stations(df, stations):
    """
    cKDTree를 사용하여 각 좌표(df)와 가장 가까운 측정소(stations)를 찾은 뒤,
    해당 측정소의 STN_ID를 df에 할당해 반환합니다.
    """
    # stations의 (longitude, latitude) 배열 생성
    station_coords = stations[['longitude', 'latitude']].values

    # cKDTree 객체 생성
    tree = cKDTree(station_coords)

    # df의 (longitude, latitude) 배열 생성
    data_coords = df[['longitude', 'latitude']].values

    # query 메서드로 각 점에 대해 가장 가까운 측정소 정보를 얻음
    distances, indices = tree.query(data_coords, k=1)
    df['STN_ID'] = stations.iloc[indices]['STN_ID'].values

    return df


def prepare_unique_requests(df):
    """
    API 요청을 효율적으로 하기 위해, time(시각)과 STN_ID(측정소)를 기준으로
    중복되지 않는 요청 목록(unique_requests)을 반환합니다.

    - 'tm' 컬럼은 'yyyymmddHH00' 형태로 포맷팅된 시간 문자열입니다.
    - 중복된 (tm, STN_ID) 조합은 제거하여 반환합니다.
    """
    df['tm'] = df['time'].dt.strftime('%Y%m%d%H00')  # 'yyyymmddHH00' 형식
    unique_requests = df[['tm', 'STN_ID']].drop_duplicates()
    return unique_requests


def get_observation_data_text(tm, stn, auth_key, help_param=0):
    """
    기상청 API를 호출하여, 특정 시각(tm)과 측정소(stn)에 대한 관측 데이터를 텍스트 형태로 반환합니다.
    - rate limit(403)이 발생하면 RateLimitExceededError를 발생시켜서, 상위 로직에서 처리를 중단하도록 합니다.
    - 기타 요청 예외는 간단히 로그를 남기고 None을 반환합니다.
    """
    url = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php'
    params = {
        'tm': tm,
        'stn': stn,
        'help': help_param,
        'authKey': auth_key
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 403:
            # Rate limit exceeded
            raise RateLimitExceededError(f"Rate limit exceeded for tm={tm}, stn={stn}")
        response.raise_for_status()  # HTTP 에러 코드(4xx, 5xx)가 있으면 예외 발생
        if not response.text.strip():
            print(f"Empty response for tm={tm}, stn={stn}")
            return None
        return response.text  # 응답 텍스트 반환
    except RateLimitExceededError as e:
        print(e)
        raise  # 상위에서 처리할 수 있도록 예외 재전달
    except requests.exceptions.RequestException as e:
        print(f"Request failed for tm={tm}, stn={stn}: {e}")
        return None


def fetch_observation_data(unique_requests, auth_key, help_param=0, max_workers=10):
    """
    ThreadPoolExecutor를 활용하여 병렬로 기상청 API를 호출하고,
    observation_data 딕셔너리에 (tm, stn)를 키, response.text를 값으로 저장한 뒤 반환합니다.

    - RateLimitExceededError가 발생하면, 즉시 스레드 풀을 종료하고 상위 레벨로 예외를 재전달합니다.
    - 일정한 인터벌(time.sleep(0.1))을 두어 서버 부하를 줄입니다.
    """
    observation_data = {}

    def fetch_and_store_text(row):
        tm = row['tm']
        stn = row['STN_ID']
        data = get_observation_data_text(tm, stn, auth_key, help_param)
        if data:
            observation_data[(tm, stn)] = data
        # Add a small delay to reduce server load
        time.sleep(0.1)
        return

    # ThreadPoolExecutor를 통해 병렬 요청 실행
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_and_store_text, row) for idx, row in unique_requests.iterrows()]
        for future in as_completed(futures):
            try:
                future.result()
            except RateLimitExceededError as e:
                # Rate Limit 초과 예외 발생 시 즉시 스레드 풀 종료
                print(f"Rate limit exceeded: {e}")
                executor.shutdown(wait=False)
                raise e  # 상위 레벨로 예외를 던져서 추가 작업 중단
            except Exception as e:
                print(f"An error occurred: {e}")

    return observation_data


def parse_observation_text(text_content):
    """
    API 응답 텍스트 중에서 유효한 데이터 라인을 추출하여
    온도(TA) 정보를 딕셔너리 형태로 반환합니다. (예: {'TA': 25.3})
    - TA(온도) 값이 없거나, 파싱에 실패하면 {'TA': np.nan}을 반환합니다.
    """
    try:
        lines = text_content.splitlines()
        data_line = None
        for line in lines:
            if line.startswith('#'):
                continue  # 주석 라인은 건너뜀
            if line.strip() == '':
                continue  # 빈 라인은 건너뜀
            data_line = line.strip()
            break  # 첫 번째 유효 데이터 라인을 찾으면 반복 종료

        if not data_line:
            print("No data line found in the response.")
            return {'TA': np.nan}

        # 공백 기준으로 필드 분리
        fields = data_line.split()

        if len(fields) < 12:
            print("Insufficient number of fields in the data line.")
            return {'TA': np.nan}

        # 'TA'는 12번째 필드(인덱스 11)
        ta = fields[11]

        # TA값이 숫자가 아니면 NaN으로 설정
        try:
            ta_float = float(ta)
        except ValueError:
            ta_float = np.nan

        return {'TA': ta_float}
    except Exception as e:
        print(f"Error parsing observation text: {e}")
        return {'TA': np.nan}


def parse_all_observations(observation_data):
    """
    observation_data에 저장된 모든 텍스트를 순회하며,
    parse_observation_text() 함수를 호출해 파싱 결과를 딕셔너리로 저장 후 반환합니다.
    - key: (tm, stn)
    - value: 예: {'TA': 25.3}
    """
    parsed_data = {}
    for key, text_content in observation_data.items():
        data = parse_observation_text(text_content)
        parsed_data[key] = data
    return parsed_data


def add_external_temp(df, parsed_data):
    """
    df의 각 행에 대해 (tm, STN_ID)를 key로 하여 parsed_data에서 TA를 찾아 'ext_temp' 컬럼에 저장합니다.
    - TA값이 없으면 NaN을 부여합니다.
    """

    def extract_TA_text(row):
        key = (row['tm'], row['STN_ID'])
        data = parsed_data.get(key, {})
        try:
            return data.get('TA', np.nan)  # TA가 없으면 np.nan 반환
        except (ValueError, TypeError):
            return np.nan

    df['ext_temp'] = df.apply(extract_TA_text, axis=1)
    return df


def save_processed_data(df, original_file):
    """
    최종적으로 처리된 df를 지정된 CSV 파일 경로(original_file)에 덮어쓰는 함수입니다.
    """
    # 저장할 컬럼 리스트 정의
    columns_tosave = ['time', 'x', 'y', 'longitude', 'latitude', 'speed', 'acceleration', 'ext_temp']

    # 기존 파일을 덮어씁니다
    df.to_csv(original_file, columns=columns_tosave, index=False)
    print(f"Data saved successfully: {original_file}")


# ----------------------------------------------------------------------------
# 2. 파일별 처리 함수 정의
# ----------------------------------------------------------------------------

def process_file(file_path, stations, auth_key, help_param=0):
    """
    단일 CSV 파일을 처리하는 핵심 로직입니다.
    1) CSV 로드
    2) 측정소 매칭
    3) API 요청을 위한 (tm, STN_ID) 추출
    4) API 호출 및 결과 텍스트 수집
    5) 텍스트 파싱
    6) 온도(ext_temp) 컬럼 추가
    7) 최종 파일 저장
    """
    print(f"Processing file: {file_path}")

    # Load data
    df = load_data(file_path)

    # Find nearest observation stations
    df = find_nearest_stations(df, stations)

    # Prepare unique API requests
    unique_requests = prepare_unique_requests(df)

    # Fetch observation data via API calls
    observation_data = fetch_observation_data(unique_requests, auth_key, help_param)

    # Parse all API responses
    parsed_data = parse_all_observations(observation_data)

    # Add external temperature data
    df = add_external_temp(df, parsed_data)

    # Overwrite the original file
    save_processed_data(df, file_path)


def load_processed_files(processed_files_path):
    """
    이미 처리된 파일 이름을 저장해둔 텍스트 파일(processed_files_path)에서
    목록을 읽어 List 형태로 반환합니다.
    """
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            processed_files = f.read().splitlines()
    else:
        processed_files = []
    return processed_files


def save_processed_file(processed_files_path, file_name):
    """
    처리된 파일 이름을 텍스트 파일(processed_files_path)에 추가로 기록합니다.
    """
    with open(processed_files_path, 'a') as f:
        f.write(file_name + '\n')


# Helper function for parallel processing
def check_file_needs_processing(file_path):
    """
    파일에 'ext_temp' 컬럼이 없거나, NaN 값이 하나라도 존재하면 True를, 그렇지 않으면 False를 반환합니다.
    """
    try:
        df = pd.read_csv(file_path)
        if 'ext_temp' not in df.columns:
            return (file_path, True)
        elif df['ext_temp'].isna().any():
            return (file_path, True)
        else:
            return (file_path, False)
    except Exception as e:
        print(f"Error reading file ({file_path}): {e}")
        return (file_path, False)


# ----------------------------------------------------------------------------
# 3. 메인 함수 정의
# ----------------------------------------------------------------------------
def main():
    # .env 파일(환경변수) 로딩
    load_dotenv()
    auth_key = os.getenv('KMA_API_KEY')
    if not auth_key:
        print("Error: KMA_API_KEY not found in environment variables.")
        return

    # 도움 파라미터(예시용)
    help_param = 0

    # 데이터 폴더 및 측정소 정보 파일 설정
    input_folder = r"D:\SamsungSTF\Processed_Data\KOTI"
    stations_file = r"D:\SamsungSTF\Data\KMA\Stations.csv"

    # 측정소 데이터 로드
    stations = load_stations(stations_file)

    # input_folder 내 모든 CSV 파일 목록을 가져옵니다.
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in the folder: {input_folder}")
        return

    print(f"Number of CSV files found: {len(csv_files)}")

    # 병렬(멀티프로세스)로 각 파일을 검사하여,
    # 'ext_temp' 컬럼이 없는 경우 혹은 NaN이 존재하는 경우만 후속 처리를 진행
    files_to_process = []
    with ProcessPoolExecutor() as executor:
        # 모든 파일을 대상으로 check_file_needs_processing 함수 제출
        future_to_file = {executor.submit(check_file_needs_processing, file): file for file in csv_files}
        # tqdm을 사용해 진행 상황을 표시
        for future in tqdm(as_completed(future_to_file), total=len(csv_files), desc="Checking files for 'ext_temp'"):
            file, needs_processing = future.result()
            if needs_processing:
                files_to_process.append(file)

    if not files_to_process:
        print("No files require processing (no missing 'ext_temp' values).")
        return

    print(f"Number of CSV files to process: {len(files_to_process)}")

    # 필요한 파일들만 순차적으로 처리
    for file in tqdm(files_to_process, desc="Processing files"):
        try:
            process_file(file, stations, auth_key, help_param)
        except RateLimitExceededError as e:
            print(f"Rate limit exceeded: {e}")
            print("Processing stopped due to rate limit.")
            break  # 더 이상의 파일 처리는 중단
        except Exception as e:
            print(f"Error processing file ({file}): {e}")

    print("All applicable files have been processed.")


if __name__ == "__main__":
    main()
