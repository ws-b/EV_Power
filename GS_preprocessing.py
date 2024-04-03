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
        # 파일의 인코딩을 감지하여 데이터를 읽음
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(100000))  # 첫 100,000 바이트를 사용하여 인코딩 감지
        encoding = result['encoding']
        return pd.read_csv(file_path, encoding=encoding, header=0)
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        # 인코딩 오류 포함, 파싱 오류가 발생한 경우 처리
        print(f"오류가 발생한 파일: {file_path}, 오류: {e}")
        return None  # 오류가 발생한 경우 None 반환

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

def process_device_folders(source_paths, destination_root):
    # 차종별 단말기 번호 리스트
    NiroEV = ['01241228149', '01241228151', '01241228153', '01241228154', '01241228155']
    Bongo3EV = ['01241228162', '01241228179', '01241248642', '01241248723', '01241248829']
    Ionic5 = [
        '01241227999', '01241228003', '01241228005', '01241228007', '01241228009', '01241228014', '01241228016',
        '01241228020', '01241228024', '01241228025', '01241228026', '01241228030', '01241228037', '01241228044',
        '01241228046', '01241228047', '01241248780', '01241248782', '01241248790', '01241248811', '01241248815',
        '01241248817', '01241248820', '01241248827', '01241364543', '01241364560', '01241364570', '01241364581',
        '01241592867', '01241592868', '01241592878', '01241592896', '01241592907', '01241597801', '01241597802',
        '01241248919'
    ]
    Ionic6 = ['01241248713', '01241592904', '01241597763', '01241597804']
    KonaEV = [
        '01241228102', '01241228122', '01241228123', '01241228156', '01241228197', '01241228203', '01241228204',
        '01241248726', '01241248727', '01241364621', '01241124056'
    ]
    Porter2EV = ['01241228144', '01241228160', '01241228177', '01241228188', '01241228192']
    EV6 = [
        '01241225206', '01241228048', '01241228049', '01241228050', '01241228051', '01241228053', '01241228054',
        '01241228055', '01241228057', '01241228059', '01241228073', '01241228075', '01241228076', '01241228082',
        '01241228084', '01241228085', '01241228086', '01241228087', '01241228090', '01241228091', '01241228092',
        '01241228094', '01241228095', '01241228097', '01241228098', '01241228099', '01241228103', '01241228104',
        '01241228106', '01241228107', '01241228114', '01241228124', '01241228132', '01241228134', '01241248679',
        '01241248818', '01241248831', '01241248833', '01241248842', '01241248843', '01241248850', '01241248860',
        '01241248876', '01241248877', '01241248882', '01241248891', '01241248892', '01241248900', '01241248903',
        '01241248908', '01241248912', '01241248913', '01241248921', '01241248924', '01241248926', '01241248927',
        '01241248929', '01241248932', '01241248933', '01241248934', '01241321943', '01241321947', '01241364554',
        '01241364575', '01241364592', '01241364627', '01241364638', '01241364714'
    ]
    GV60 = ['01241228108', '01241228130', '01241228131', '01241228136', '01241228137', '01241228138']

    # 단말기 번호와 차종명의 매핑 생성
    device_to_car = {}
    for car_type, devices in [
        ("NiroEV", NiroEV), ("Bongo3EV", Bongo3EV), ("Ionic5", Ionic5), ("Ionic6", Ionic6),
        ("KonaEV", KonaEV), ("Porter2EV", Porter2EV), ("EV6", EV6), ("GV60", GV60)
    ]:
        for device in devices:
            device_to_car[device] = car_type

    # 각 원본 경로에 대해 처리
    for source_path in source_paths:
        for year_month in os.listdir(source_path):
            year_month_path = os.path.join(source_path, year_month)
            if os.path.isdir(year_month_path):
                for device_number in os.listdir(year_month_path):
                    if device_number in device_to_car:
                        device_number_path = os.path.join(year_month_path, device_number)
                        car_type = device_to_car[device_number]
                        if 'bms_altitude' in source_path:
                            destination_device_folder = device_number + '_GPS'
                        else:
                            destination_device_folder = device_number
                        destination_path = os.path.join(destination_root, car_type, destination_device_folder,
                                                        year_month)
                        os.makedirs(destination_path, exist_ok=True)
                        for file in os.listdir(device_number_path):
                            shutil.move(os.path.join(device_number_path, file), destination_path)
                            print(f'Moved {file} to {destination_path}')

def process_bms_files(start_path, save_path):
    total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)])

    with tqdm(total=total_folders, desc="진행 상황", unit="folder") as pbar:
        for root, dirs, files in os.walk(start_path):
            if not dirs:
                filtered_files = [file for file in files if 'bms' in file and file.endswith('.csv') and 'altitude' not in file]
                filtered_files.sort()
                dfs = []
                device_no, year_month = None, None
                for file in filtered_files:
                    file_path = os.path.join(root, file)
                    df = read_file_with_detected_encoding(file_path)

                    if df is not None:  # df가 None이 아닐 때만 처리
                        df = df.loc[:, ~df.columns.str.contains('Unnamed')]  # Unnamed 컬럼 제거
                        df = df.iloc[1:][::-1]
                        dfs.append(df)

                        if device_no is None or year_month is None:
                            parts = file.split('_')
                            device_no = parts[1]  # 단말기 번호
                            date_parts = parts[2].split('-')
                            year_month = '-'.join(date_parts[:2])  # 연월 (YYYY-MM 형식)
                    else:
                        continue

                    if device_no is None or year_month is None:
                        parts = file.split('_')
                        device_no = parts[1]  # 단말기 번호
                        date_parts = parts[2].split('-')
                        year_month = '-'.join(date_parts[:2])  # 연월 (YYYY-MM 형식)
                        print(device_no, year_month)

                if dfs and device_no and year_month:
                    combined_df = pd.concat(dfs, ignore_index=True)

                    # calculate time and speed changes
                    combined_df['time'] = combined_df['time'].str.strip()
                    combined_df['time'] = pd.to_datetime(df['time'], format='%y-%m-%d %H:%M:%S')
                    t = pd.to_datetime(combined_df['time'], format='%y-%m-%d %H:%M:%S')
                    t_diff = t.diff().dt.total_seconds()
                    combined_df['time_diff'] = t_diff
                    combined_df['speed'] = combined_df[
                                               'emobility_spd'] * 0.27778  # convert speed to m/s if originally in km/h

                    # Calculate speed difference using central differentiation
                    combined_df['spd_diff'] = combined_df['speed'].rolling(window=3, center=True).apply(
                        lambda x: x[2] - x[0],
                        raw=True) / 2

                    # calculate acceleration using the speed difference and time difference
                    combined_df['acceleration'] = combined_df['spd_diff'] / combined_df['time_diff']

                    # Handling edge cases for acceleration (first and last elements)
                    combined_df.at[0, 'acceleration'] = (combined_df.at[1, 'speed'] - combined_df.at[0, 'speed']) / \
                                                        combined_df.at[1, 'time_diff']
                    combined_df.at[len(combined_df) - 1, 'acceleration'] = (combined_df.at[
                                                                                len(combined_df) - 1, 'speed'] -
                                                                            combined_df.at[
                                                                                len(combined_df) - 2, 'speed']) / \
                                                                           combined_df.at[
                                                                               len(combined_df) - 1, 'time_diff']

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

                    # 저장 경로 생성
                    parts = root.split(os.sep)  # os.sep은 시스템에 따라 적절한 경로 구분자를 사용합니다.
                    vehicle_type = parts[-3]  # 차종 정보
                    device_no = parts[-2].split('_')[0]  # 단말기 번호

                    save_folder = os.path.join(save_path, vehicle_type)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)  # 해당 경로가 없다면 생성

                    output_file_name = f'bms_{device_no}_{year_month}.csv'
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
        altitude_files = [f for f in files if f.startswith('altitude') and f.endswith('.csv')]
        bms_files = [f for f in files if f.startswith('bms') and f.endswith('.csv') and 'altitude' not in f]

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

def process_bms_altitude_files(start_path, save_path):
    total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)])

    with tqdm(total=total_folders, desc="진행 상황", unit="folder") as pbar:
        for root, dirs, files in os.walk(start_path):
            if not dirs:
                filtered_files = [file for file in files if file.startswith('bms_altitude_') and file.endswith('.csv')]
                filtered_files.sort()
                dfs = []
                device_no, year_month = None, None
                for file in filtered_files:
                    file_path = os.path.join(root, file)
                    df = read_file_with_detected_encoding(file_path)

                    if df is not None:  # df가 None이 아닐 때만 처리
                        df = df.loc[:, ~df.columns.str.contains('Unnamed')]  # Unnamed 컬럼 제거
                        df = df.iloc[1:][::-1]
                        dfs.append(df)

                        if device_no is None or year_month is None:
                            parts = file.split('_')
                            device_no = parts[2]  # 단말기 번호
                            date_parts = parts[3].split('-')
                            year_month = '-'.join(date_parts[:2])  # 연월 (YYYY-MM 형식)
                            print(device_no, year_month)
                    else:
                        # df가 None인 경우, 즉 파일 읽기에서 오류가 발생한 경우, 해당 파일은 건너뜀
                        continue

                    if device_no is None or year_month is None:
                        parts = file.split('_')
                        device_no = parts[2]  # 단말기 번호
                        date_parts = parts[3].split('-')
                        year_month = '-'.join(date_parts[:2])  # 연월 (YYYY-MM 형식)
                        print(device_no, year_month)

                if dfs and device_no and year_month:
                    combined_df = pd.concat(dfs, ignore_index=True)

                    # calculate time and speed changes
                    combined_df['time'] = combined_df['time'].str.strip()
                    combined_df['time'] = pd.to_datetime(df['time'], format='%y-%m-%d %H:%M:%S')
                    t = pd.to_datetime(combined_df['time'], format='%y-%m-%d %H:%M:%S')
                    t_diff = t.diff().dt.total_seconds()
                    combined_df['time_diff'] = t_diff
                    combined_df['speed'] = combined_df[
                                               'emobility_spd'] * 0.27778  # convert speed to m/s if originally in km/h

                    # Calculate speed difference using central differentiation
                    combined_df['spd_diff'] = combined_df['speed'].rolling(window=3, center=True).apply(
                        lambda x: x[2] - x[0], raw=True) / 2

                    # calculate acceleration using the speed difference and time difference
                    combined_df['acceleration'] = combined_df['spd_diff'] / combined_df['time_diff']

                    # Handling edge cases for acceleration (first and last elements)
                    combined_df.at[0, 'acceleration'] = (combined_df.at[1, 'speed'] - combined_df.at[0, 'speed']) / \
                                                        combined_df.at[1, 'time_diff']
                    combined_df.at[len(combined_df) - 1, 'acceleration'] = (combined_df.at[
                                                                                len(combined_df) - 1, 'speed'] -
                                                                            combined_df.at[
                                                                                len(combined_df) - 2, 'speed']) / \
                                                                           combined_df.at[
                                                                               len(combined_df) - 1, 'time_diff']

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

                    # 저장 경로 생성
                    parts = root.split(os.sep)  # os.sep은 시스템에 따라 적절한 경로 구분자를 사용합니다.
                    vehicle_type = parts[-3]  # 차종 정보
                    device_no = parts[-2].split('_')[0]  # 단말기 번호

                    save_folder = os.path.join(save_path, vehicle_type)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)  # 해당 경로가 없다면 생성

                    output_file_name = f'bms_altitude_{device_no}_{year_month}.csv'
                    data_save.to_csv(os.path.join(save_folder, output_file_name), index=False)
                pbar.update(1)

    print("모든 폴더의 파일 처리가 완료되었습니다.")
def process_files_trip_by_trip(file_lists, folder_path, save_path):
    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)

        # Load CSV file into a pandas DataFrame
        data = pd.read_csv(file_path)

        cut = []

        # Parse Trip by cable connection status
        if data.loc[0, 'chrg_cable_conn'] == 0:
            cut.append(0)
        for i in range(len(data)-1):
            if data.loc[i, 'chrg_cable_conn'] != data.loc[i+1, 'chrg_cable_conn']:
                cut.append(i+1)
        if data.loc[len(data)-1, 'chrg_cable_conn'] == 0:
            cut.append(len(data)-1)

        # Parse Trip by Time difference
        cut_time = pd.Timedelta(seconds=300)  # 300sec 이상 차이 날 경우 다른 Trip으로 인식
        data['time'] = pd.to_datetime(data['time'], format="%Y-%m-%d %H:%M:%S")  # Convert 'time' column to datetime
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
                filename = f"{file[:11]}-{month:02}-trip-{trip_counter}.csv"
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
                filename = f"{file[:11]}-{month:02}-trip-{trip_counter}.csv"
                trip.to_csv(os.path.join(save_path, filename), index=False)
    print("Done")

def check_trip_conditions(trip):
    # If trip dataframe is empty, return False
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
        return False  # Trip does not meet the conditions

    return True

def process_files_combined(file_lists, folder_path, save_path):
    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)

        # Load CSV file into a pandas DataFrame
        df = pd.read_csv(file_path, dtype={'device_no': str, 'measured_month': str})

        # calculate time and speed changes
        df['time'] = df['time'].str.strip()
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        t = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds()
        df['time_diff'] = t_diff
        df['speed'] = df['emobility_spd'] * 0.27778  # convert speed to m/s if originally in km/h

        # Calculate speed difference using central differentiation
        df['spd_diff'] = df['speed'].rolling(window=3, center=True).apply(lambda x: x[2] - x[0], raw=True) / 2

        # calculate acceleration using the speed difference and time difference
        df['acceleration'] = df['spd_diff'] / df['time_diff']

        # Handling edge cases for acceleration (first and last elements)
        df.at[0, 'acceleration'] = (df.at[1, 'speed'] - df.at[0, 'speed']) / df.at[1, 'time_diff']
        df.at[len(df) - 1, 'acceleration'] = (df.at[len(df) - 1, 'speed'] - df.at[len(df) - 2, 'speed']) / df.at[len(df) - 1, 'time_diff']

        # replace NaN values with 0 or fill with desired values
        df['acceleration'] = df['acceleration'].fillna(0)

        # additional calculations...
        df['Power_IV'] = df['pack_volt'] * df['pack_current']
        if 'altitude' in df.columns:
            # 'delta altitude' 열 추가
            df['delta altitude'] = df['altitude'].diff()
            # merge selected columns into a single DataFrame
            data_save = df[['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh', 'chrg_cable_conn', 'altitude', 'pack_current', 'pack_volt', 'Power_IV']].copy()
        else:
            # merge selected columns into a single DataFrame
            data_save = df[['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh', 'chrg_cable_conn', 'pack_current', 'pack_volt', 'Power_IV']].copy()

        # save as a CSV file
        device_no = df['device_no'].iloc[0].replace(' ', '')
        if not device_no.startswith('0'):
            device_no = '0' + device_no

        file_name = f"{device_no}{'-' + df['measured_month'].iloc[0][-2:].replace(' ', '')}.csv"
        full_path = os.path.join(save_path, file_name)

        data_save.to_csv(full_path, index=False)

    print('Done')