import os
import pandas as pd

win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\한국에너지공과대학교_샘플데이터\speed-acc'
win_save_path = r'G:\공유 드라이브\Battery Software Lab\Data\한국에너지공과대학교_샘플데이터\trip_by_trip'

folder_path = os.path.normpath(win_folder_path)
save_path = os.path.normpath(win_save_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

for file in file_lists:
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