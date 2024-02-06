import pandas as pd
from datetime import datetime, timedelta
import glob
import os

# 최상위 폴더 경로 설정
base_path = '/Users/wsong/Downloads/test_case/'

def match_closest_bms_time(altitude_time):
    # bms와 altitude 시간 차이 계산
    time_diff = bms_df['time'] - altitude_time
    # 가장 작은 시간 차이를 가진 bms 인덱스 찾기
    closest_bms_index = time_diff.abs().idxmin()
    # 가장 작은 시간 차이가 3초 이상인지 확인
    if time_diff.abs().min() > timedelta(seconds=3):
        return None  # 3초 이상 차이나면 None 반환
    return closest_bms_index

# 차종 폴더 탐색
vehicle_folders = glob.glob(os.path.join(base_path, '*/'))

for vehicle_folder in vehicle_folders:
    # 각 차종 폴더 내의 단말기 번호 폴더 탐색
    device_folders = glob.glob(os.path.join(vehicle_folder, '*/'))

    for device_folder in device_folders:
        # 각 단말기 폴더 내의 'altitude'와 'bms' 파일 찾기
        altitude_files = sorted(glob.glob(os.path.join(device_folder, '*altitude*.csv')))
        bms_files = [file for file in sorted(glob.glob(os.path.join(device_folder, '*bms*.csv')))
                     if not os.path.basename(file).startswith('alt_bms_')]

        # 파일 쌍별로 병합 작업 수행
        for altitude_file, bms_file in zip(altitude_files, bms_files):
            # 파일 읽기
            altitude_df = pd.read_csv(altitude_file)
            bms_df = pd.read_csv(bms_file)

            # 시간 형식 정의 및 변환
            bms_df['time'] = pd.to_datetime(bms_df['time'], format="%y-%m-%d %H:%M:%S")
            altitude_df['time'] = pd.to_datetime(altitude_df['time'], format="%Y-%m-%d %H:%M:%S")

            # bms_df에 'altitude' 열 추가 및 초기화
            bms_df['altitude'] = pd.NA

            # 각 altitude 데이터에 대해 가장 근접한 bms 행 찾기 및 altitude 값 매핑
            for altitude_index in altitude_df.index:
                bms_index = match_closest_bms_time(altitude_df.at[altitude_index, 'time'])
                if bms_index is not None:  # 3초 이내에 근접한 경우에만 매핑
                    bms_df.at[bms_index, 'altitude'] = altitude_df.at[altitude_index, 'altitude']

            # 'altitude' 열을 float 타입으로 변환
            bms_df['altitude'] = pd.to_numeric(bms_df['altitude'], errors='coerce')

            # 'time_diff' 열을 추가하여 시간 간격 계산
            bms_df['time_diff'] = bms_df['time'].diff().dt.total_seconds()

            # 누락된 데이터 구간 탐지를 위한 임계값 설정 (예: 60초)
            missing_threshold = 60

            # 첫 번째 유효한 'altitude' 값으로 누락된 값을 채우고 선형 보간
            for idx in range(len(bms_df)):
                if bms_df.iloc[idx]['time_diff'] > missing_threshold:
                    # 다음 유효한 'altitude' 값 탐색
                    next_valid_altitude = bms_df.iloc[idx:]['altitude'].first_valid_index()
                    if pd.notnull(next_valid_altitude):
                        # 유효한 'altitude' 값이 존재하면 누락된 값을 첫 번째 유효한 값으로 채움
                        bms_df.at[idx, 'altitude'] = bms_df.at[next_valid_altitude, 'altitude']
                else:
                    pass

            # 변환 후 선형 보간 적용
            bms_df['altitude'] = bms_df['altitude'].interpolate(method='linear', limit_direction='both')

            # 사용한 열 제거
            bms_df.drop(columns=['time_diff'], inplace=True)

            # 병합된 데이터 저장
            merged_filename = f"alt_bms_{os.path.basename(device_folder.strip('/'))}_{altitude_file.split('_')[-1]}"
            merged_filepath = os.path.join(device_folder, merged_filename)
            bms_df.to_csv(merged_filepath, index=False)