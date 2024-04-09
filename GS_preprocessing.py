import os
import pandas as pd
import numpy as np
import shutil
import chardet
from datetime import datetime, timedelta
from tqdm import tqdm

def get_file_list(folder_path, file_extension='.csv'):
    file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(file_extension)]
    file_lists.sort()
    return file_lists

def read_file_with_detected_encoding(file_path):
    try:
        # 우선 C 엔진으로 시도
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # C 엔진 실패 시 ISO-8859-1 인코딩으로 재시도
        try:
            return pd.read_csv(file_path, encoding='iso-8859-1')
        except Exception as e:
            # ISO-8859-1도 실패할 경우 Python 엔진으로 시도
            try:
                return pd.read_csv(file_path, encoding='utf-8', engine='python')
            except Exception as e:
                print(f"Failed to read file {file_path} with Python engine due to: {e}")
                return None
def process_device_folders(source_paths, destination_root):
    for year_month in os.listdir(source_paths):
        year_month_path = os.path.join(source_paths, year_month)
        if os.path.isdir(year_month_path):  # 년-월 폴더 확인
            for device_number in os.listdir(year_month_path):
                device_number_path = os.path.join(year_month_path, device_number)
                if os.path.isdir(device_number_path):  # 단말기 번호 폴더 확인
                    # 대상 폴더 경로 생성 (단말기번호/년-월)
                    destination_path = os.path.join(destination_root, device_number, year_month)
                    os.makedirs(destination_path, exist_ok=True)  # 폴더가 없다면 생성

                    # 파일 이동
                    for file in os.listdir(device_number_path):
                        source_file_path = os.path.join(device_number_path, file)
                        destination_file_path = os.path.join(destination_path, file)
                        shutil.move(source_file_path, destination_file_path)  # 파일 이동
                        print(f"Moved {file} to {destination_path}")

def process_bms_files(start_path, save_path, device_vehicle_mapping):
    total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)])

    with tqdm(total=total_folders, desc="Processing folders", unit="folder") as pbar:
        for root, dirs, files in os.walk(start_path):
            if not dirs:
                filtered_files = [f for f in files if f.endswith('.csv') and 'bms' in f and 'altitude' not in f and '01241228107' in f]
                filtered_files.sort()
                dfs = []
                device_no, year_month = None, None
                for file in filtered_files:
                    file_path = os.path.join(root, file)
                    df = read_file_with_detected_encoding(file_path)
                    if df is not None:
                        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
                        df = df.iloc[::-1].reset_index(drop=True)  # 행 순서를 역순으로 뒤집고 인덱스를 리셋
                        dfs.append(df)

                        if device_no is None or year_month is None:
                            parts = file_path.split(os.sep)
                            file_name = parts[-1]
                            name_parts = file_name.split('_')
                            device_no = name_parts[1]
                            date_parts = name_parts[2].split('-')
                            year_month = '-'.join(date_parts[:2])
                    else:
                        continue

                if dfs and device_no and year_month:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    print(f"Processing file: {file_path}")  # 현재 처리 중인 파일 경로 출력

                    # calculate time and speed changes
                    combined_df['time'] = combined_df['time'].str.strip()
                    try:
                        t = pd.to_datetime(combined_df['time'], format='%y-%m-%d %H:%M:%S')
                    except ValueError:
                        try:
                            t = pd.to_datetime(combined_df['time'], format='%Y-%m-%d %H:%M:%S')
                        except ValueError as e:
                            print(f"Date format error: {e}")
                            continue
                    t_diff = t.diff().dt.total_seconds()
                    combined_df['time_diff'] = t_diff
                    combined_df['speed'] = combined_df['emobility_spd'] * 0.27778  # convert speed to m/s if originally in km/h

                    # Calculate speed difference using central differentiation
                    combined_df['spd_diff'] = combined_df['speed'].rolling(window=3, center=True).apply(
                        lambda x: x[2] - x[0], raw=True) / 2

                    # calculate acceleration using the speed difference and time difference
                    combined_df['acceleration'] = combined_df['spd_diff'] / combined_df['time_diff']

                    # Handling edge cases for acceleration (first and last elements)
                    combined_df.at[0, 'acceleration'] = (combined_df.at[1, 'speed'] - combined_df.at[0, 'speed']) / \
                                                        combined_df.at[1, 'time_diff']
                    combined_df.at[len(combined_df) - 1, 'acceleration'] = (combined_df.at[len(combined_df) - 1, 'speed'] - combined_df.at[len(combined_df) - 2, 'speed']) / \
                                                                           combined_df.at[len(combined_df) - 1, 'time_diff']

                    # replace NaN values with 0 or fill with desired values
                    combined_df['acceleration'] = combined_df['acceleration'].fillna(0)

                    # additional calculations...
                    combined_df['Power_IV'] = combined_df['pack_volt'] * combined_df['pack_current']
                    if 'altitude' in combined_df.columns:
                        # 'delta altitude' 열 추가
                        combined_df['delta altitude'] = combined_df['altitude'].diff()
                        # merge selected columns into a single DataFrame
                        data_save = combined_df[
                            ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh', 'chrg_cable_conn',
                             'altitude', 'pack_volt', 'pack_current', 'Power_IV']].copy()
                    else:
                        # merge selected columns into a single DataFrame
                        data_save = combined_df[
                            ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh', 'chrg_cable_conn', 'pack_volt',
                             'pack_current', 'Power_IV']].copy()

                    vehicle_type = device_vehicle_mapping.get(device_no, 'Unknown')
                    save_folder = os.path.join(save_path, vehicle_type)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    output_file_name = f"bms_{device_no}_{year_month}.csv"
                    data_save.to_csv(os.path.join(save_folder, output_file_name), index=False)

                pbar.update(1)

    print("모든 폴더의 파일 처리가 완료되었습니다.")

def process_gps_files(start_path, save_path):
    total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)])

    with tqdm(total=total_folders, desc="진행 상황", unit="folder") as pbar:
        for root, dirs, files in os.walk(start_path):
            if not dirs:
                filtered_files = [file for file in files if 'gps' in file and file.endswith('.csv') and 'bms' not in file]
                filtered_files.sort()  # 파일 이름으로 정렬
                dfs = []  # 각 파일의 DataFrame을 저장할 리스트
                device_no, year_month = None, None
                for file in filtered_files:
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path, header=0, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, header=0, encoding='cp949')

                    # 'Unnamed'으로 시작하는 컬럼과 컬럼명이 비어있는 컬럼 제거
                    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
                    df = df.loc[:, df.columns != '']  # 빈 컬럼명 제거

                    # 첫 행을 제외하고 역순으로 정렬
                    df = df.iloc[1:][::-1]
                    dfs.append(df)

                    if device_no is None or year_month is None:
                        parts = file.split('_')
                        device_no = parts[1]  # 단말기 번호
                        date_parts = parts[2].split('-')
                        year_month = '-'.join(date_parts[:2])  # 연월 (YYYY-MM 형식)
                        print(device_no, year_month)

                if dfs and device_no and year_month:
                    combined_df = pd.concat(dfs, ignore_index=True)

                    # 저장 경로 생성
                    parts = root.split(os.sep)  # os.sep은 시스템에 따라 적절한 경로 구분자를 사용합니다.
                    vehicle_type = parts[-3]  # 차종 정보
                    device_no = parts[-2].split('_')[0]  # 단말기 번호

                    save_folder = os.path.join(save_path, vehicle_type)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)  # 해당 경로가 없다면 생성

                    output_file_name = f'gps_{device_no}_{year_month}.csv'
                    combined_df.to_csv(os.path.join(save_folder, output_file_name), index=False)
                pbar.update(1)

    print("모든 폴더의 파일 처리가 완료되었습니다.")

def merge_bms_gps(start_path):
    def extract_device_date_from_filename(filename):
        # 파일명에서 "단말기번호_연-월" 부분을 추출
        parts = filename.split('_')
        device_date = '_'.join(parts[1:2])
        return device_date
    def match_closest_bms_time(bms_df, altitude_time):
        time_diff = bms_df['time'] - altitude_time
        closest_bms_index = time_diff.abs().idxmin()
        if time_diff.abs().min() > timedelta(seconds=3):
            return None
        return closest_bms_index

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
                bms_df['time'] = pd.to_datetime(bms_df['time'], format="%Y-%m-%d %H:%M:%S")
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

                # # 고도에 대한 선형 보간
                # bms_df['altitude'] = pd.to_numeric(bms_df['altitude'], errors='coerce').interpolate(method='linear', limit_direction='both')
                # # 위도와 경도에 대한 forward fill 방식 적용
                # bms_df['lat'] = bms_df['lat'].ffill()
                # bms_df['lng'] = bms_df['lng'].ffill()

                return bms_df
        return None
    files_to_process = []

    # 모든 파일을 순회하며 매칭되는 파일 쌍을 찾음
    for root, _, files in os.walk(start_path):
        altitude_files = [file for file in files if file.startswith('gps') and file.endswith('.csv')]
        bms_files = [file for file in files if file.startswith('bms') and file.endswith('.csv') and 'altitude' not in file]

        for altitude_file in altitude_files:
            matching_bms_file = find_matching_bms_file(altitude_file, bms_files)
            if matching_bms_file:
                files_to_process.append((os.path.join(root, altitude_file), os.path.join(root, matching_bms_file)))

    total_files = len(files_to_process)

    with tqdm(total=total_files, desc="Processing", unit="file") as pbar:
        for altitude_file, bms_file in files_to_process:
            # 이전과 동일한 데이터 처리 로직을 사용하여 파일 처리
            merged_df = process_files([altitude_file], [bms_file])
            if merged_df is not None:
                output_file_path = os.path.join(os.path.dirname(altitude_file),
                                                f"bms_altitude_{os.path.basename(altitude_file)[9:]}")
                merged_df.to_csv(output_file_path, index=False)
            pbar.update(1)

    print("All folders processed.")

    # 원본 파일 삭제 로직 추가
    for altitude_file, bms_file in files_to_process:
        try:
            os.remove(altitude_file)
            os.remove(bms_file)
            print(f"Deleted: {altitude_file} and {bms_file}")
        except Exception as e:
            print(f"Error deleting file: {e}")

    print("Original files deleted.")

def process_bms_altitude_files(start_path, save_path, device_vehicle_mapping):
    total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)])

    with tqdm(total=total_folders, desc="Processing folders", unit="folder") as pbar:
        for root, dirs, files in os.walk(start_path):
            if not dirs:
                filtered_files = [f for f in files if f.endswith('.csv') and 'bms' in f and 'altitude' in f]
                filtered_files.sort()
                dfs = []
                device_no, year_month = None, None
                for file in filtered_files:
                    file_path = os.path.join(root, file)
                    df = read_file_with_detected_encoding(file_path)
                    if df is not None:
                        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
                        df = df.iloc[::-1].reset_index(drop=True)  # 행 순서를 역순으로 뒤집고 인덱스를 리셋
                        dfs.append(df)

                        if device_no is None or year_month is None:
                            parts = file_path.split(os.sep)
                            file_name = parts[-1]
                            name_parts = file_name.split('_')
                            device_no = name_parts[2]
                            date_parts = name_parts[3].split('-')
                            year_month = '-'.join(date_parts[:2])
                    else:
                        continue

                if dfs and device_no and year_month:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    print(f"Processing file: {file_path}")  # 현재 처리 중인 파일 경로 출력

                    # calculate time and speed changes
                    combined_df['time'] = combined_df['time'].str.strip()
                    try:
                        t = pd.to_datetime(combined_df['time'], format='%y-%m-%d %H:%M:%S')
                    except ValueError:
                        try:
                            t = pd.to_datetime(combined_df['time'], format='%Y-%m-%d %H:%M:%S')
                        except ValueError as e:
                            print(f"Date format error: {e}")
                            continue
                    t_diff = t.diff().dt.total_seconds()
                    combined_df['time_diff'] = t_diff
                    combined_df['speed'] = combined_df['emobility_spd'] * 0.27778  # convert speed to m/s if originally in km/h

                    # Calculate speed difference using central differentiation
                    combined_df['spd_diff'] = combined_df['speed'].rolling(window=3, center=True).apply(
                        lambda x: x[2] - x[0], raw=True) / 2

                    # calculate acceleration using the speed difference and time difference
                    combined_df['acceleration'] = combined_df['spd_diff'] / combined_df['time_diff']

                    # Handling edge cases for acceleration (first and last elements)
                    combined_df.at[0, 'acceleration'] = (combined_df.at[1, 'speed'] - combined_df.at[0, 'speed']) / \
                                                        combined_df.at[1, 'time_diff']
                    combined_df.at[len(combined_df) - 1, 'acceleration'] = (combined_df.at[len(combined_df) - 1, 'speed'] - combined_df.at[len(combined_df) - 2, 'speed']) / \
                                                                           combined_df.at[len(combined_df) - 1, 'time_diff']

                    # replace NaN values with 0 or fill with desired values
                    combined_df['acceleration'] = combined_df['acceleration'].fillna(0)

                    # additional calculations...
                    combined_df['Power_IV'] = combined_df['pack_volt'] * combined_df['pack_current']
                    if 'altitude' in combined_df.columns:
                        # 'delta altitude' 열 추가
                        combined_df['delta altitude'] = combined_df['altitude'].diff()
                        # merge selected columns into a single DataFrame
                        data_save = combined_df[
                            ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh', 'chrg_cable_conn',
                             'altitude', 'pack_volt', 'pack_current', 'Power_IV']].copy()
                    else:
                        # merge selected columns into a single DataFrame
                        data_save = combined_df[
                            ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh', 'chrg_cable_conn', 'pack_volt',
                             'pack_current', 'Power_IV']].copy()

                    vehicle_type = device_vehicle_mapping.get(device_no, 'Unknown')
                    save_folder = os.path.join(save_path, vehicle_type)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    output_file_name = f"bms_altitude_{device_no}_{year_month}.csv"
                    data_save.to_csv(os.path.join(save_folder, output_file_name), index=False)

                pbar.update(1)

    print("모든 폴더의 파일 처리가 완료되었습니다.")
def process_files_trip_by_trip(file_lists, start_path, save_path):
    total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)])

    with tqdm(total=total_folders, desc="Processing folders", unit="folder") as pbar:
        for root, dirs, files in os.walk(start_path):
            if not dirs:
                all_files = [f for f in files if f.endswith('.csv')]
                all_files.sort()

                for file in all_files:
                    file_path = os.path.join(root, file)
                    print(file_path.split(os.sep)[-1])
                    data = pd.read_csv(file_path)
                    if 'altitude' in data.columns:
                        parts = file_path.split(os.sep)
                        file_name = parts[-1]
                        name_parts = file_name.split('_')
                        device_no = name_parts[2]
                        year_month = name_parts[3][:7]
                    else:
                        parts = file_path.split(os.sep)
                        file_name = parts[-1]
                        name_parts = file_name.split('_')
                        device_no = name_parts[1]
                        year_month = name_parts[2][:7]

                    cut = []

                    # Parse Trip by cable connection status
                    if data.loc[0, 'chrg_cable_conn'] == 0:
                        cut.append(0)
                    for i in range(len(data) - 1):
                        if data.loc[i, 'chrg_cable_conn'] != data.loc[i + 1, 'chrg_cable_conn']:
                            cut.append(i + 1)
                    if data.loc[len(data) - 1, 'chrg_cable_conn'] == 0:
                        cut.append(len(data) - 1)

                    # Parse Trip by Time difference
                    cut_time = pd.Timedelta(seconds=300)  # 300sec 이상 차이 날 경우 다른 Trip으로 인식
                    try:
                        data['time'] = pd.to_datetime(data['time'], format='%y-%m-%d %H:%M:%S')
                    except ValueError:
                        try:
                            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
                        except ValueError as e:
                            print(f"Date format error: {e}")
                            continue

                    for i in range(len(data) - 1):
                        if data.loc[i + 1, 'time'] - data.loc[i, 'time'] > cut_time:
                            cut.append(i + 1)

                    cut = list(set(cut))
                    cut.sort()

                    trip_counter = 1  # Start trip number from 1 for each file
                    for i in range(len(cut) - 1):
                        if data.loc[cut[i], 'chrg_cable_conn'] == 0:
                            trip = data.loc[cut[i]:cut[i + 1] - 1, :]

                            # Check if the trip meets the conditions from the first function
                            if not check_trip_conditions(trip):
                                continue

                            # Formulate the filename based on the given rule
                            month = trip['time'].iloc[0].month
                            if 'altitude' in data.columns:
                                filename = f"bms_altitude_{device_no}-{year_month}-trip-{trip_counter}.csv"
                            else:
                                filename = f"bms_{device_no}-{year_month}-trip-{trip_counter}.csv"
                            # Save to file
                            trip.to_csv(os.path.join(save_path, filename), index=False)
                            trip_counter += 1

                    # for the last trip
                    trip = data.loc[cut[-1]:, :]

                    # Check if the last trip meets the conditions from the first function
                    if check_trip_conditions(trip):
                        duration = trip['time'].iloc[-1] - trip['time'].iloc[0]
                        if duration >= pd.Timedelta(minutes=5) and data.loc[cut[-1], 'chrg_cable_conn'] == 0:
                            month = trip['time'].iloc[0].month
                            if 'altitude' in data.columns:
                                filename = f"bms_altitude_{device_no}-{year_month}-trip-{trip_counter}.csv"
                            else:
                                filename = f"bms_{device_no}-{year_month}-trip-{trip_counter}.csv"
                            trip.to_csv(os.path.join(save_path, filename), index=False)
    print("Done")

def check_trip_conditions(trip):
    if trip.empty:
        return False

    # Calculate conditions from the first function for the trip
    v = trip['speed']
    t = pd.to_datetime(trip['time'], format='%Y-%m-%d %H:%M:%S')
    t_diff = t.diff().dt.total_seconds().fillna(0)
    v = np.array(v)
    distance = v * t_diff
    total_distance = distance.cumsum().iloc[-1]
    time_range = t.iloc[-1] - t.iloc[0]
    data_power = trip['Power_IV']
    data_power = np.array(data_power)
    data_energy = data_power * t_diff / 3600 / 1000
    data_energy_cumulative = data_energy.cumsum().iloc[-1]

    # Check if any of the conditions are met for the trip
    time_limit = 300
    distance_limit = 3000
    Energy_limit = 1.0
    if time_range.total_seconds() < time_limit or total_distance < distance_limit or data_energy_cumulative < Energy_limit or (trip['acceleration'].abs() > 9.8).any():
        return False

    return True