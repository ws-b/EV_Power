import os
import pandas as pd
from tqdm import tqdm
import chardet
from datetime import datetime, timedelta

# 시작 디렉토리 설정
start_path = '/Volumes/Data/test_case'

def match_closest_bms_time(bms_df, altitude_time):
    time_diff = bms_df['time'] - altitude_time
    closest_bms_index = time_diff.abs().idxmin()
    if time_diff.abs().min() > timedelta(seconds=3):
        return None
    return closest_bms_index


def read_file_with_detected_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(100000))
        encoding = result['encoding']
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def extract_device_date_from_filename(filename):
    # 파일명에서 "단말기번호_연-월" 부분을 추출
    parts = filename.split('_')
    if len(parts) > 2:
        device_date = '_'.join(parts[1:3])  # 단말기번호_연-월 부분
        return device_date
    return None

def find_matching_bms_file(altitude_file, bms_files):
    device_date = extract_device_date_from_filename(altitude_file)
    for bms_file in bms_files:
        if device_date in bms_file:
            return bms_file
    return None

def process_files(altitude_files, bms_files):
    for altitude_file, bms_file in zip(altitude_files, bms_files):
        altitude_df = read_file_with_detected_encoding(altitude_file)
        bms_df = read_file_with_detected_encoding(bms_file)

        if altitude_df is not None and bms_df is not None:
            bms_df['time'] = pd.to_datetime(bms_df['time'], format="%y-%m-%d %H:%M:%S")
            altitude_df['time'] = pd.to_datetime(altitude_df['time'], format="%Y-%m-%d %H:%M:%S")

            # 초기에 NA로 설정
            bms_df['altitude'] = pd.NA
            bms_df['lat'] = pd.NA
            bms_df['lng'] = pd.NA

            for index in altitude_df.index:
                closest_bms_index = match_closest_bms_time(bms_df, altitude_df.at[index, 'time'])
                if closest_bms_index is not None:
                    bms_df.at[closest_bms_index, 'altitude'] = altitude_df.at[index, 'altitude']
                    bms_df.at[closest_bms_index, 'lat'] = altitude_df.at[index, 'lat']
                    bms_df.at[closest_bms_index, 'lng'] = altitude_df.at[index, 'lng']

            # 고도에 대한 선형 보간
            bms_df['altitude'] = pd.to_numeric(bms_df['altitude'], errors='coerce').interpolate(method='linear', limit_direction='both')
            # 위도와 경도에 대한 forward fill 방식 적용
            bms_df['lat'] = bms_df['lat'].ffill()
            bms_df['lng'] = bms_df['lng'].ffill()

            return bms_df
    return None

files_to_process = []

# 모든 파일을 순회하며 매칭되는 파일 쌍을 찾음
for root, _, files in os.walk(start_path):
    # 'alt_bms'로 시작하는 파일을 제외하고 리스트를 생성
    altitude_files = [f for f in files if 'altitude' in f and f.endswith('.csv')]
    bms_files = [f for f in files if 'bms' in f and f.endswith('.csv')]

    for altitude_file in altitude_files:
        matching_bms_file = find_matching_bms_file(altitude_file, bms_files)
        if matching_bms_file:
            files_to_process.append((os.path.join(root, altitude_file), os.path.join(root, matching_bms_file)))

total_files = len(files_to_process)

with tqdm(total=total_files, desc="Processing", unit="file") as pbar:
    for altitude_file, bms_file in files_to_process:
        # 이전과 동일한 데이터 처리 로직을 사용하여 파일 처리
        merged_df = process_files([altitude_file], [bms_file])  # 수정된 process_files 함수 사용
        if merged_df is not None:
            output_file_path = os.path.join(os.path.dirname(altitude_file), f"merged_{os.path.basename(altitude_file)[9:]}")
            merged_df.to_csv(output_file_path, index=False)
        pbar.update(1)


print("All folders processed.")