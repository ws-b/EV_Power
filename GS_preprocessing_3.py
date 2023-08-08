import os
import pandas as pd
from tqdm import tqdm
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
