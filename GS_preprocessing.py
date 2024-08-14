import os
import glob
import pandas as pd
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
def load_data_by_vehicle(folder_path, vehicle_dict, selected_car):
    vehicle_files = {}
    if selected_car not in vehicle_dict:
        print(f"Selected vehicle '{selected_car}' not found in vehicle_dict.")
        return vehicle_files

    ids = vehicle_dict[selected_car]
    all_files = []
    for vid in ids:
        patterns = [
            os.path.join(folder_path, f"**/bms_{vid}-*"),
            os.path.join(folder_path, f"**/bms_altitude_{vid}-*")
        ]
        for pattern in patterns:
            all_files += glob.glob(pattern, recursive=True)
    vehicle_files[selected_car] = all_files

    return vehicle_files

def process_device_folders(source_paths, destination_root, altitude=False):
    for root, dirs, files in os.walk(source_paths):
        if not dirs:  # No subdirectories means this is a leaf directory
            if altitude:
                filtered_files = [f for f in files if f.endswith('.csv') and 'bms' in f and 'altitude' in f]
            else:
                filtered_files = [f for f in files if f.endswith('.csv') and 'bms' in f and 'altitude' not in f]

            filtered_files.sort()
            device_no, year_month = None, None

            for file in filtered_files:
                file_path = os.path.join(root, file)
                parts = file_path.split(os.sep)
                file_name = parts[-1]
                name_parts = file_name.split('_')
                device_no = name_parts[1] if not altitude else name_parts[2]
                date_parts = name_parts[2].split('-') if not altitude else name_parts[3].split('-')
                year_month = '-'.join(date_parts[:2])

                save_folder = os.path.join(destination_root, device_no, year_month)
                os.makedirs(save_folder, exist_ok=True)  # If the folder does not exist, create it

                destination_file_path = os.path.join(save_folder, file)

                shutil.move(file_path, destination_file_path)
                print(f"Moved {file} to {destination_file_path}")

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

def process_files(start_path, save_path, vehicle_type, altitude=False):
    total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)])

    with tqdm(total=total_folders, desc="Processing folders", unit="folder") as pbar:
        with ProcessPoolExecutor() as executor:
            futures = []
            for root, dirs, files in os.walk(start_path):
                if not dirs:  # Only process leaf folders
                    futures.append(
                        executor.submit(process_folder, root, files, save_path, vehicle_type, altitude))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
                finally:
                    pbar.update(1)

    print("모든 폴더의 파일 처리가 완료되었습니다.")

def process_folder(root, files, save_path, vehicle_type, altitude):
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

        vehicle_model = vehicle_type.get(device_no, 'Unknown')
        save_folder = os.path.join(save_path, vehicle_model)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        output_file_name = f"{'bms_altitude' if altitude else 'bms'}_{device_no}_{year_month}.csv"
        output_file_path = os.path.join(save_folder, output_file_name)

        if os.path.exists(output_file_path):
            print(f"File {output_file_name} already exists in {save_folder}. Skipping...")
            return

        df = read_file_with_detected_encoding(file_path)
        if df is not None:
            df = df.loc[:, ~df.columns.str.contains('Unnamed')]
            df = df.drop_duplicates(subset='time')
            df = df.iloc[::-1].reset_index(drop=True)
            dfs.append(df)

    if dfs and device_no and year_month and not os.path.exists(output_file_path):
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Processing file: {output_file_path}")

        combined_df['time'] = combined_df['time'].str.strip()
        date_formats = ['%y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S']
        for date_format in date_formats:
            try:
                t = pd.to_datetime(combined_df['time'], format=date_format)
                break
            except ValueError as e:
                print(f"Date format error: {e}")
                continue
        else:
            print(f"Date format error for file {output_file_path}")
            return

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
        combined_df['Power_data'] = combined_df['pack_volt'] * combined_df['pack_current']
        if 'altitude' in combined_df.columns:
            combined_df['delta altitude'] = combined_df['altitude'].diff()
            data_save = combined_df[
                ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh', 'chrg_cable_conn',
                 'altitude', 'pack_volt', 'pack_current', 'Power_data']].copy()
        else:
            data_save = combined_df[
                ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh', 'chrg_cable_conn',
                 'pack_volt', 'pack_current', 'Power_data']].copy()

        data_save.to_csv(output_file_path, index=False)

def process_files_trip_by_trip(start_path, save_path):
    # Calculate the total number of CSV files
    csv_files = [os.path.join(root, file)
                 for root, _, files in os.walk(start_path)
                 for file in files if file.endswith('.csv')]
    total_files = len(csv_files)

    # Progress bar setup
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        # Process files in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_wrapper, file_path, save_path) for file_path in csv_files]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'File {future_to_file[future]} generated an exception: {exc}')
                finally:
                    pbar.update(1)

    print("Processing complete")

def process_wrapper(file_path, save_path):
    try:
        process_single_file(file_path, save_path)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        raise

def process_single_file(file_path, save_path):
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
    existing_files = [f for f in os.listdir(save_path) if
                      f.startswith(altitude_file_pattern) or f.startswith(non_altitude_file_pattern)]

    if existing_files:
        print(f"Files {device_no} and {year_month} already exist. Skipping all related files.")
        return

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
            return

    for i in range(len(data) - 1):
        if data.loc[i + 1, 'time'] - data.loc[i, 'time'] > cut_time:
            cut.append(i + 1)

    cut = list(set(cut))
    cut.sort()

    if not cut:
        print(f"No cuts found in file: {file_path}")
        return None

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

    if cut:
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

def check_trip_conditions(trip):
    if trip.empty:
        return False

    if (trip['acceleration'] > 9.0).any():
        return False

    v = trip['speed']
    date_formats = ['%y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S']

    for date_format in date_formats:
        try:
            t = pd.to_datetime(trip['time'], format=date_format)
            break
        except ValueError:
            continue
    else:
        print("Date format error in trip conditions")
        return False

    t_diff = t.diff().dt.total_seconds().fillna(0)
    distance = (v * t_diff).cumsum().iloc[-1]

    time_limit = 300
    distance_limit = 3000
    energy_limit = 1.0

    if (t.iloc[-1] - t.iloc[0]).total_seconds() < time_limit or distance < distance_limit:
        return False

    data_energy = (trip['Power_data'] * t_diff / 3600 / 1000).cumsum().iloc[-1]
    if data_energy < energy_limit:
        return False

    zero_speed_duration = 0
    for i in range(len(trip) - 1):
        if trip['speed'].iloc[i] == 0:
            zero_speed_duration += (t.iloc[i + 1] - t.iloc[i]).total_seconds()
            if zero_speed_duration >= 300:
                return False
        else:
            zero_speed_duration = 0

    return True
