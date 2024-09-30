import os
import glob
import pandas as pd
import requests
from datetime import datetime
from tqdm import tqdm
from functools import lru_cache
import math

# 관측소 정보 불러오기 (KMA에서 제공하는 관측소 목록 CSV 파일 필요)
# 예: 'observation_stations.csv' 파일에 'station_id', 'station_name', 'lat', 'lon' 컬럼이 포함되어 있어야 합니다.
OBSERVATION_STATIONS_CSV = 'observation_stations.csv'


def load_observation_stations(csv_path):
    """
    관측소 목록을 CSV 파일에서 불러옵니다.
    CSV 파일은 최소한 'station_id', 'station_name', 'lat', 'lon' 컬럼을 포함해야 합니다.
    """
    df = pd.read_csv(csv_path)
    required_columns = {'station_id', 'station_name', 'lat', 'lon'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV 파일은 {required_columns} 컬럼을 포함해야 합니다.")
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    두 지점 간의 거리를 계산합니다. (단위: km)
    """
    R = 6371  # 지구 반지름 (km)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def find_nearest_station(stations_df, lat, lon):
    """
    주어진 위도와 경도에 가장 가까운 관측소를 찾습니다.
    """
    stations_df['distance'] = stations_df.apply(
        lambda row: haversine_distance(lat, lon, row['lat'], row['lon']), axis=1
    )
    nearest_station = stations_df.loc[stations_df['distance'].idxmin()]
    return nearest_station['station_id'], nearest_station['station_name']


def read_all_csv(folder_path):
    """
    지정된 폴더 내 모든 CSV 파일을 읽어와 하나의 DataFrame으로 합칩니다.
    """
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"'{folder_path}' 경로에 CSV 파일이 없습니다.")

    df_list = []
    for file in csv_files:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
            print(f"'{file}' 파일을 성공적으로 읽었습니다.")
        except Exception as e:
            print(f"'{file}' 파일을 읽는 중 오류 발생: {e}")

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"모든 CSV 파일을 합쳐 총 {combined_df.shape[0]}개의 행을 가진 데이터프레임을 생성했습니다.")
    return combined_df


def get_temperature_kma(station_id, date_time, api_key):
    """
    KMA Open API를 사용하여 특정 관측소의 특정 시간에 대한 온도 데이터를 가져옵니다.
    - station_id: 관측소 ID
    - date_time: datetime 객체
    - api_key: 발급받은 KMA API 인증키
    """
    # 실제 API 엔드포인트와 파라미터는 KMA Open API 문서를 참고하여 수정하세요.
    # 아래는 예시입니다.
    url = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"

    params = {
        'serviceKey': api_key,
        'pageNo': 1,
        'numOfRows': 1,
        'dataType': 'JSON',
        'dataCd': 'ASOS',
        'dateCd': 'HR',  # 시간별 데이터
        'startDt': date_time.strftime('%Y%m%d'),
        'startHh': date_time.strftime('%H'),
        'endDt': date_time.strftime('%Y%m%d'),
        'endHh': date_time.strftime('%H'),
        'stnIds': station_id
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            # JSON 구조는 API 문서에 따라 다릅니다. 예시:
            items = data.get('response', {}).get('body', {}).get('items', {})
            if 'item' in items:
                # 'ta'는 기온을 나타내는 예시 필드입니다. 실제 필드명을 확인하세요.
                if isinstance(items['item'], list):
                    temp = items['item'][0].get('ta')  # 기온
                else:
                    temp = items['item'].get('ta')
                return temp
            else:
                print(f"데이터 없음: station_id={station_id}, date_time={date_time}")
                return None
        else:
            print(f"API 요청 실패: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"요청 예외 발생: {e}")
        return None


@lru_cache(maxsize=10000)
def get_temperature_cached_kma(station_id, date_time_str, api_key):
    """
    캐시된 KMA 온도 데이터 가져오기 함수.
    - station_id: 관측소 ID
    - date_time_str: 'YYYY-MM-DD HH:MM:SS' 형식의 시간 문자열
    - api_key: KMA API 인증키
    """
    date_time = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    return get_temperature_kma(station_id, date_time, api_key)


def add_temperature_data_kma(df, stations_df, api_key):
    """
    DataFrame에 'ext_temp' 컬럼을 추가하고, 각 행에 해당하는 온도 데이터를 저장합니다.
    """
    df['ext_temp'] = None  # 'ext_temp' 컬럼 초기화

    temperatures = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="온도 데이터 가져오는 중"):
        time_str = row.get('time')
        lng = row.get('lng')
        lat = row.get('lat')

        # 데이터 유효성 검사
        if pd.isnull(time_str) or pd.isnull(lat) or pd.isnull(lng):
            print(f"누락된 데이터 at index {idx}: time={time_str}, lat={lat}, lng={lng}")
            temperatures.append(None)
            continue

        # 시간 문자열을 datetime 객체로 변환
        try:
            date_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"시간 형식 오류: {time_str}")
            temperatures.append(None)
            continue

        # 가장 가까운 관측소 찾기
        try:
            station_id, station_name = find_nearest_station(stations_df, lat, lng)
        except Exception as e:
            print(f"관측소 찾기 오류 at index {idx}: {e}")
            temperatures.append(None)
            continue

        # 온도 데이터 가져오기
        try:
            temp = get_temperature_cached_kma(station_id, time_str, api_key)
            temperatures.append(temp)
        except Exception as e:
            print(f"온도 데이터 가져오기 오류 at index {idx}: {e}")
            temperatures.append(None)

    df['ext_temp
