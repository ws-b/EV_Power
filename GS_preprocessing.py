import os
import pandas as pd
import numpy as np
import shutil
import chardet
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta
from tqdm import tqdm

def get_file_list(folder_path, file_extension='.csv'):
    file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(file_extension)]
    file_lists.sort()
    return file_lists

def read_file_with_detected_encoding(file_path):
    try:
        # First, try to read the file with the C engine and UTF-8 encoding
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If C engine fails, try ISO-8859-1 encoding
        try:
            return pd.read_csv(file_path, encoding='iso-8859-1')
        except Exception as e:
            # If both C and ISO-8859-1 engines fail, try to detect the encoding
            try:
                return pd.read_csv(file_path, encoding='utf-8', engine='python')
            except Exception as e:
                print(f"Failed to read file {file_path} with Python engine due to: {e}")
                return None
            
def process_device_folders(source_paths, destination_root):
    for year_month in os.listdir(source_paths):
        year_month_path = os.path.join(source_paths, year_month)
        if os.path.isdir(year_month_path):  # check year-month folder
            for device_number in os.listdir(year_month_path):
                device_number_path = os.path.join(year_month_path, device_number)
                if os.path.isdir(device_number_path): # check device number folder
                    # Create destination folder
                    destination_path = os.path.join(destination_root, device_number, year_month)
                    os.makedirs(destination_path, exist_ok=True)  # If the folder does not exist, create it

                    # Move files
                    for file in os.listdir(device_number_path):
                        source_file_path = os.path.join(device_number_path, file)
                        destination_file_path = os.path.join(destination_path, file)
                        shutil.move(source_file_path, destination_file_path) 
                        print(f"Moved {file} to {destination_path}")

def process_bms_files(start_path, save_path, device_vehicle_mapping):
    total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)])

    with tqdm(total=total_folders, desc="Processing folders", unit="folder") as pbar:
        for root, dirs, files in os.walk(start_path):
            if not dirs:
                filtered_files = [f for f in files if f.endswith('.csv') and 'bms' in f and 'altitude' not in f]
                filtered_files.sort()
                dfs = []
                device_no, year_month = None, None
                for file in filtered_files:
                    file_path = os.path.join(root, file)
                    
                    # Extract device number and year-month before processing the file
                    parts = file_path.split(os.sep)
                    file_name = parts[-1]
                    name_parts = file_name.split('_')
                    device_no = name_parts[1]
                    date_parts = name_parts[2].split('-')
                    year_month = '-'.join(date_parts[:2])

                    vehicle_type = device_vehicle_mapping.get(device_no, 'Unknown')
                    save_folder = os.path.join(save_path, vehicle_type)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    output_file_name = f"bms_{device_no}_{year_month}.csv"
                    output_file_path = os.path.join(save_folder, output_file_name)

                    if os.path.exists(output_file_path):
                        print(f"File {output_file_name} already exists in {save_folder}. Skipping...")
                        break

                    df = read_file_with_detected_encoding(file_path)
                    if df is not None:
                        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
                        df = df.drop_duplicates(subset='time')
                        df = df.iloc[::-1].reset_index(drop=True)
                        dfs.append(df)

                if dfs and device_no and year_month and not os.path.exists(output_file_path):
                    combined_df = pd.concat(dfs, ignore_index=True)
                    print(f"Processing file: {file_path}")

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
                            ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh','cumul_pw_chrgd', 'cumul_pw_dischrgd', 'chrg_cable_conn',
                            'altitude', 'pack_volt', 'pack_current', 'Power_IV']].copy()
                    else:
                        # merge selected columns into a single DataFrame
                        data_save = combined_df[
                            ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh','cumul_pw_chrgd', 'cumul_pw_dischrgd', 'chrg_cable_conn',
                            'pack_volt', 'pack_current', 'Power_IV']].copy()

                    data_save.to_csv(output_file_path, index=False)

                pbar.update(1)

    print("모든 폴더의 파일 처리가 완료되었습니다.")



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
                    
                    # Extract device number and year-month before processing the file
                    parts = file_path.split(os.sep)
                    file_name = parts[-1]
                    name_parts = file_name.split('_')
                    device_no = name_parts[2]
                    date_parts = name_parts[3].split('-')
                    year_month = '-'.join(date_parts[:2])

                    vehicle_type = device_vehicle_mapping.get(device_no, 'Unknown')
                    save_folder = os.path.join(save_path, vehicle_type)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    output_file_name = f"bms_altitude_{device_no}_{year_month}.csv"
                    output_file_path = os.path.join(save_folder, output_file_name)

                    if os.path.exists(output_file_path):
                        print(f"File {output_file_name} already exists in {save_folder}. Skipping...")
                        break

                    df = read_file_with_detected_encoding(file_path)
                    if df is not None:
                        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
                        df = df.drop_duplicates(subset='time')
                        df = df.iloc[::-1].reset_index(drop=True)
                        dfs.append(df)

                if dfs and device_no and year_month and not os.path.exists(output_file_path):
                    combined_df = pd.concat(dfs, ignore_index=True)
                    print(f"Processing file: {file_path}")

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

                    # Calculate speed difference using standard differentiation
                    combined_df['spd_diff'] = combined_df['speed'].diff()
                    combined_df['spd_diff'].iloc[0] = combined_df['speed'].iloc[1] - combined_df['speed'].iloc[0]  # First element
                    combined_df['spd_diff'].iloc[-1] = combined_df['speed'].iloc[-1] - combined_df['speed'].iloc[-2]  # Last element

                    # calculate acceleration using the speed difference and time difference
                    combined_df['acceleration'] = combined_df['spd_diff'] / combined_df['time_diff']

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

                    data_save.to_csv(output_file_path, index=False)

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

                    # Check if files for the given device_no and year_month already exist
                    altitude_file_pattern = f"bms_altitude_{device_no}-{year_month}-trip-"
                    non_altitude_file_pattern = f"bms_{device_no}-{year_month}-trip-"
                    existing_files = [f for f in os.listdir(save_path) if f.startswith(altitude_file_pattern) or f.startswith(non_altitude_file_pattern)]
                    
                    if existing_files:
                        print(f"Files {device_no} and {year_month} already exist. Skipping all related files.")
                        continue

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
                            os.makedirs(save_path, exist_ok=True)
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
                            print(f"Files {device_no} and {year_month} successfully processed.")
                            os.makedirs(save_path, exist_ok=True)
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