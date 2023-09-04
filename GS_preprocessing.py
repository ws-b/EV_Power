import os
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def get_file_list(folder_path, file_extension='.csv'):
    file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(file_extension)]
    file_lists.sort()
    return file_lists
def parse_spacebar(file_lists, folder_path, save_path):
    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as infile:  # Open the file
            reader = csv.reader(infile, delimiter='|')

            data = []
            last_line = None
            for i, row in enumerate(reader):
                if i == 0 or i == 2:  # skip first and third row
                    continue
                if last_line is not None:
                    data.append(last_line)
                last_line = row

            # 첫 번째 행에서 각 열의 공백을 제거합니다.
            if data:  # Make sure data is not empty
                data[0] = [col.strip() for col in data[0]]

            # ','를 구분자로 사용해 출력 파일을 작성합니다.
            with open(os.path.join(save_path, file[:-4] + "_parsed.csv"), 'w', newline='') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                writer.writerows(data)
    print("Done!")

def merge_csv_files(file_lists, folder_path):
    # 11자리 숫자를 키로 하여 파일들을 그룹화합니다.
    grouped_files = defaultdict(list)
    for file in tqdm(file_lists):
        key = file[:11]
        grouped_files[key].append(file)

    merged_dataframes = {}

    for key, files in grouped_files.items():
        # 각 그룹의 CSV 파일을 읽어들여 하나의 데이터프레임 리스트에 저장합니다.
        list_of_dfs = [pd.read_csv(os.path.join(folder_path, f)) for f in files]

        # 모든 데이터프레임을 하나로 병합합니다.
        merged_df = pd.concat(list_of_dfs, ignore_index=True)

        # Append the merged dataframe to the list
        merged_dataframes[key] = merged_df

    return merged_dataframes

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
        data['time'] = pd.to_datetime(data['time'])  # Convert 'time' column to datetime
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
    bms_power = trip['Power_IV']
    bms_power = np.array(bms_power)
    data_energy = bms_power * t_diff / 3600 / 1000
    data_energy_cumulative = data_energy.cumsum().iloc[-1]

    # Check if any of the conditions are met for the trip
    time_limit = 300
    distance_limit = 3000
    Energy_limit = 1.0
    if time_range.total_seconds() < time_limit or total_distance < distance_limit or data_energy_cumulative < Energy_limit:
        return False  # Trip does not meet the conditions
    return True

def process_files_combined(file_lists, folder_path, save_path):
    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)

        # Load CSV file into a pandas DataFrame
        df = pd.read_csv(file_path, dtype={'device_no': str, 'measured_month': str})

        # reverse the DataFrame based on the index
        df = df[::-1]

        # calculate time and speed changes
        df['time'] = df['time'].str.strip()
        df['time'] = pd.to_datetime(df['time'], format='%y-%m-%d %H:%M:%S')
        t = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds()
        df['time_diff'] = t_diff
        df['speed'] = df['emobility_spd'] * 0.27778
        df['spd_diff'] = df['speed'].diff()

        # calculate acceleration
        df['acceleration'] = df['spd_diff'] / df['time_diff']

        # replace NaN values with 0 or fill with desired values
        df['acceleration'] = df['acceleration'].replace(np.nan, 0)

        # merge selected columns into a single DataFrame
        df['Power_IV'] = df['pack_volt'] * df['pack_current']

        # merge selected columns into a single DataFrame
        data_save = df[['time', 'speed', 'acceleration',
                        'ext_temp', 'int_temp', 'soc', 'soh','chrg_cable_conn', 'pack_current', 'pack_volt', 'Power_IV']].copy()

        # save as a CSV file
        device_no = df['device_no'].iloc[0].replace(' ', '')
        if not device_no.startswith('0'):
            device_no = '0' + device_no

        file_name = f"{device_no}{'-0' + df['measured_month'].iloc[0][-2:].replace(' ', '')}.csv"
        full_path = os.path.join(save_path, file_name)

        data_save.to_csv(full_path, index=False)

    print('Done')