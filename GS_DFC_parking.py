import os
import pandas as pd
import shutil
from tqdm import tqdm


def filter_and_copy_fully_charged_files(source_dir, destination_dir, time_gap_minutes=1):
    """
    주차 세그먼트 파일 중에서 'chrg_cable_conn'이 1이고,
    해당 시점의 앞뒤 시간 간격이 1분 이상인 파일들을 필터링하여 복사합니다.

    Args:
        source_dir (str): 주차 세그먼트 CSV 파일들이 저장된 디렉토리 경로.
        destination_dir (str): 조건을 만족하는 파일들을 복사할 디렉토리 경로.
        time_gap_minutes (int, optional): 시간 간격 기준 (기본값: 1분).

    Returns:
        list: 복사된 파일들의 경로 리스트.
    """
    # 복사할 디렉토리가 존재하지 않으면 생성
    os.makedirs(destination_dir, exist_ok=True)

    # 소스 디렉토리 내의 모든 CSV 파일 목록 가져오기
    csv_files = [os.path.join(source_dir, file)
                 for file in os.listdir(source_dir)
                 if file.endswith('.csv')]

    selected_files = []

    for file in tqdm(csv_files, desc="Filtering and copying files"):
        try:
            data = pd.read_csv(file)
        except Exception as e:
            print(f"파일 {file}을 읽는 중 오류 발생: {e}")
            continue

        # 필수 열 존재 여부 확인
        required_columns = ['chrg_cable_conn', 'speed', 'time', 'soc']
        if not all(col in data.columns for col in required_columns):
            print(f"파일 {file}에 필수 열 중 하나 이상이 없습니다. 건너뜁니다.")
            continue

        # 'time' 열을 datetime 형식으로 변환
        try:
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        except Exception as e:
            print(f"파일 {file}에서 'time' 열을 변환하는 중 오류 발생: {e}")
            continue

        # 변환된 'time' 열에 NaT가 있는지 확인
        if data['time'].isnull().any():
            print(f"파일 {file}에 'time' 열에 잘못된 형식의 데이터가 포함되어 있습니다. 건너뜁니다.")
            continue

        # 'chrg_cable_conn'이 1인 행 찾기
        charging_events = data[data['chrg_cable_conn'] == 1]

        if charging_events.empty:
            continue  # 'chrg_cable_conn'이 1인 경우가 없으면 다음 파일로

        # 각 충전 이벤트에 대해 앞뒤 시간 간격 확인
        is_fully_charged = False
        for idx in charging_events.index:
            current_time = data.at[idx, 'time']

            # 이전 행의 시간 확인
            if idx == 0:
                prev_time_gap = pd.Timedelta(minutes=1000)  # 충분히 큰 값으로 설정
            else:
                prev_time = data.at[idx - 1, 'time']
                prev_time_gap = current_time - prev_time

            # 다음 행의 시간 확인
            if idx == len(data) - 1:
                next_time_gap = pd.Timedelta(minutes=1000)  # 충분히 큰 값으로 설정
            else:
                next_time = data.at[idx + 1, 'time']
                next_time_gap = next_time - current_time

            # 시간 간격이 기준 이상인지 확인
            if prev_time_gap >= pd.Timedelta(minutes=time_gap_minutes) and next_time_gap >= pd.Timedelta(
                    minutes=time_gap_minutes):
                is_fully_charged = True
                break  # 하나의 조건 만족 시 파일 선택

        if is_fully_charged:
            try:
                shutil.copy(file, destination_dir)
                selected_files.append(file)
            except Exception as e:
                print(f"파일 {file}을 복사하는 중 오류 발생: {e}")
                continue

    return selected_files


# **사용 예시**

# 주차 세그먼트 파일들이 저장된 디렉토리 경로
source_directory = 'D:\SamsungSTF\Processed_Data\Parking'  # 실제 경로로 변경

# 조건을 만족하는 파일들을 복사할 디렉토리 경로
destination_directory = r'D:\SamsungSTF\Processed_Data\Parking\fully_charged'  # 실제 경로로 변경

# 파일 필터링 및 복사
copied_files = filter_and_copy_fully_charged_files(source_directory, destination_directory)

print(f"조건을 만족하는 파일 수: {len(copied_files)}")
for file in copied_files:
    print(f"복사된 파일: {file}")
