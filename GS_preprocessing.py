import os
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
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
        data_save = df[['time', 'speed', 'acceleration', 'trip_chrg_pw', 'trip_dischrg_pw', 'pack_current', 'pack_volt',
                        'chrg_cable_conn', 'ext_temp', 'int_temp', 'soc', 'soh', 'Power_IV']].copy()

        # save as a CSV file
        data_save.to_csv(os.path.join(save_path,
                                      f"{df['device_no'].iloc[0].replace(' ', '')}{'-0' + df['measured_month'].iloc[0][-2:].replace(' ', '')}.csv"),
                         index=False)

    print('Done')
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

        # Parse Trip by Trip Charge & Trip Discharge
        for i in range(len(data) - 1):
            if data.loc[i + 1, 'trip_dischrg_pw'] - data.loc[i, 'trip_dischrg_pw'] != 0 and data.loc[i + 1, 'trip_chrg_pw'] - data.loc[i, 'trip_chrg_pw'] != 0  and data.loc[i+1, 'trip_dischrg_pw'] == 0 and data.loc[i+1, 'trip_chrg_pw'] == 0:
                cut.append(i + 1)

        # Parse Trip by Trip Discharge difference
        for i in range(len(data) - 1):
            if abs(data.loc[i + 1, 'trip_dischrg_pw'] - data.loc[i, 'trip_dischrg_pw']) > 0.5:
                cut.append(i + 1)

        cut = list(set(cut))
        cut.sort()

        trip_counter = 1  # Start trip number from 1 for each file
        for i in range(len(cut)-1):
            if data.loc[cut[i], 'chrg_cable_conn'] == 0:
                trip = data.loc[cut[i]:cut[i+1]-1, :]

                # Check the duration of the trip
                duration = trip['time'].iloc[-1] - trip['time'].iloc[0]
                if duration >= pd.Timedelta(minutes=5):
                    # Save to file
                    trip.to_csv(f"{save_path}/{file[:-4]}-trip-{trip_counter}.csv", index=False)
                    trip_counter += 1

            # for the last trip
            trip = data.loc[cut[-1]:, :]
            duration = trip['time'].iloc[-1] - trip['time'].iloc[0]
            if duration >= pd.Timedelta(minutes=5) and data.loc[cut[-1], 'chrg_cable_conn'] == 0:
                trip.to_csv(f"{save_path}/{file[:-4]}-trip-{trip_counter}.csv", index=False)
    print("Done")

def process_files_all_trips(file_lists, folder_path, save_path):
    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)

        # Load CSV file into a pandas DataFrame
        data = pd.read_csv(file_path)
        cut = []
        all_trips = []

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

        # Parse Trip by Trip Charge & Trip Discharge
        for i in range(len(data) - 1):
            if data.loc[i + 1, 'trip_dischrg_pw'] - data.loc[i, 'trip_dischrg_pw'] != 0 and data.loc[i + 1, 'trip_chrg_pw'] - data.loc[i, 'trip_chrg_pw'] != 0  and data.loc[i+1, 'trip_dischrg_pw'] == 0 and data.loc[i+1, 'trip_chrg_pw'] == 0:
                cut.append(i + 1)

        # Parse Trip by Trip Discharge difference
        for i in range(len(data) - 1):
            if abs(data.loc[i + 1, 'trip_dischrg_pw'] - data.loc[i, 'trip_dischrg_pw']) > 0.5:
                cut.append(i + 1)

        cut = list(set(cut))
        cut.sort()

        trip_counter = 1  # Start trip number from 1 for each file
        for i in range(len(cut) - 1):
            if data.loc[cut[i], 'chrg_cable_conn'] == 0:
                trip = data.loc[cut[i]:cut[i + 1] - 1, :]

                # Check the duration of the trip
                duration = trip['time'].iloc[-1] - trip['time'].iloc[0]
                if duration >= pd.Timedelta(minutes=5):
                    all_trips.append(trip)

        # For the last trip
        trip = data.loc[cut[-1]:, :]
        duration = trip['time'].iloc[-1] - trip['time'].iloc[0]
        if duration >= pd.Timedelta(minutes=5) and data.loc[cut[-1], 'chrg_cable_conn'] == 0:
            all_trips.append(trip)

        combined_trips = pd.concat(all_trips, ignore_index=True)
        combined_trips.to_csv(os.path.join(save_path, f"{file[:-4]}_all_trips.csv"), index=False)
       # Combine all trips into a single DataFrame and save to file

    print("Done")
