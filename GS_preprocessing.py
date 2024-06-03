import os
import pandas as pd
import numpy as np
import shutil
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

def interpolate_outliers(df, flags, window=8):
    df_interpolated = df.copy()
    df_interpolated['flag'] = False  # Add flag column to indicate interpolation

    # Handle consecutive zeros in speed
    zero_flags = (df['speed'] == 0)

    # Mark sequences of consecutive zeros
    consecutive_zero_counts = zero_flags.astype(int).groupby(zero_flags.ne(zero_flags.shift()).cumsum()).cumsum()
    zero_flags_consecutive = consecutive_zero_counts >= 3

    combined_flags = flags | zero_flags_consecutive

    for i in range(len(df)):
        if combined_flags[i]:
            # Determine the window for surrounding values
            start = max(i - window, 0)
            end = min(i + window + 1, len(df))

            # Calculate the mean and standard deviation of the surrounding values, ignoring zeros
            surrounding_values = df['speed'][start:end][df['speed'][start:end] != 0]
            surrounding_mean = surrounding_values.mean()

            # Interpolate the value
            df_interpolated.loc[i, 'speed'] = surrounding_mean
            df_interpolated.loc[i, 'flag'] = True  # Mark as interpolated

    return df_interpolated

def process_files(start_path, save_path, device_vehicle_mapping, altitude=False):
    total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)])

    with tqdm(total=total_folders, desc="Processing folders", unit="folder") as pbar:
        for root, dirs, files in os.walk(start_path):
            if not dirs:
                if altitude:
                    filtered_files = [f for f in files if f.endswith('.csv') and 'bms' in f and 'altitude' in f]
                else:
                    filtered_files = [f for f in files if f.endswith('.csv') and 'bms' in f and 'altitude' not in f]
                filtered_files.sort()
                dfs = []
                device_no, year_month = None, None
                for file in filtered_files:
                    file_path = os.path.join(root, file)

                    parts = file_path.split(os.sep)
                    file_name = parts[-1]
                    name_parts = file_name.split('_')
                    device_no = name_parts[1] if not altitude else name_parts[2]
                    date_parts = name_parts[2].split('-') if not altitude else name_parts[3].split('-')
                    year_month = '-'.join(date_parts[:2])

                    vehicle_type = device_vehicle_mapping.get(device_no, 'Unknown')
                    save_folder = os.path.join(save_path, vehicle_type)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    output_file_name = f"{'bms_altitude' if altitude else 'bms'}_{device_no}_{year_month}.csv"
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
                    combined_df['speed'] = combined_df['emobility_spd'] * 0.27778

                    # Calculate acceleration using forward and backward differences
                    combined_df['acceleration'] = combined_df['speed'].diff() / combined_df['time_diff']

                    # Handle first and last row separately to avoid NaN values
                    if len(combined_df) > 1:
                        combined_df.at[0, 'acceleration'] = (combined_df.at[1, 'speed'] - combined_df.at[0, 'speed']) / \
                                                            combined_df.at[1, 'time_diff']
                        combined_df.at[len(combined_df) - 1, 'acceleration'] = (combined_df.at[
                                                                                    len(combined_df) - 1, 'speed'] -
                                                                                combined_df.at[
                                                                                    len(combined_df) - 2, 'speed']) / \
                                                                               combined_df.at[
                                                                                   len(combined_df) - 1, 'time_diff']

                    combined_df['acceleration'] = combined_df['acceleration'].fillna(0)

                    # Flagging acceleration spikes
                    acceleration_threshold = 9.0
                    flags = combined_df['acceleration'].abs() > acceleration_threshold

                    # Interpolating speed values for flagged rows
                    combined_df = interpolate_outliers(combined_df, flags)
                    # Only recalculate the acceleration for flagged rows and their surrounding rows (8 before and 8 after)
                    recalc_indices = flags[flags].index
                    for idx in recalc_indices:
                        start = max(idx - 8, 0)
                        end = min(idx + 8 + 1, len(combined_df))
                        combined_df.loc[start:end, 'speed_diff'] = combined_df.loc[start:end, 'speed'].diff()
                        combined_df.loc[start:end, 'acceleration'] = combined_df.loc[start:end, 'speed_diff'] / combined_df.loc[start:end, 'time_diff']

                    # Handle first and last row separately to avoid NaN values
                    if len(combined_df) > 1:
                        combined_df.at[0, 'acceleration'] = (combined_df.at[1, 'speed'] - combined_df.at[0, 'speed']) / \
                                                            combined_df.at[1, 'time_diff']
                        combined_df.at[len(combined_df) - 1, 'acceleration'] = (combined_df.at[
                                                                                    len(combined_df) - 1, 'speed'] -
                                                                                combined_df.at[
                                                                                    len(combined_df) - 2, 'speed']) / \
                                                                               combined_df.at[
                                                                                   len(combined_df) - 1, 'time_diff']

                    combined_df['acceleration'] = combined_df['acceleration'].fillna(0)

                    combined_df['Power_IV'] = combined_df['pack_volt'] * combined_df['pack_current']
                    if 'altitude' in combined_df.columns:
                        combined_df['delta altitude'] = combined_df['altitude'].diff()
                        data_save = combined_df[
                            ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh', 'chrg_cable_conn',
                             'altitude', 'pack_volt', 'pack_current', 'Power_IV', 'flag']].copy()
                    else:
                        data_save = combined_df[
                            ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh', 'chrg_cable_conn',
                             'pack_volt', 'pack_current', 'Power_IV', 'flag']].copy()

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

    time_limit = 300
    distance_limit = 3000
    Energy_limit = 1.0
    if time_range.total_seconds() < time_limit or total_distance < distance_limit or data_energy_cumulative < Energy_limit:
        return False

    # Check for segments where speed is 0 for 5 minutes or more
    zero_speed_duration = 0
    for i in range(len(trip) - 1):
        if trip['speed'].iloc[i] == 0:
            zero_speed_duration += (t.iloc[i + 1] - t.iloc[i]).total_seconds()
            if zero_speed_duration >= time_limit:
                return False
        else:
            zero_speed_duration = 0

    return True