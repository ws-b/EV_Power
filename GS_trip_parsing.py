import os
import pandas as pd

# 파일이 들어있는 폴더 경로
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\speed-acc\\'
mac_folder_path = '/Users/woojin/Documents/켄텍 자료/삼성미래과제/한국에너지공과대학교_샘플데이터/speed-acc/'
win_save_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\trip_by_trip\\'
mac_save_path = '/Users/woojin/Documents/켄텍 자료/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip/'

folder_path = mac_folder_path
save_path = mac_save_path

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
for file_list in file_lists:
    file = open(folder_path + file_list, "r")

    # Load CSV file into a pandas DataFrame
    data = pd.read_csv(file)
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
    for i in range(len(cut)-1):
        if data.loc[cut[i], 'chrg_cable_conn'] == 0:
            trip = data.loc[cut[i]:cut[i+1]-1, :]

            # Check the duration of the trip
            duration = trip['time'].iloc[-1] - trip['time'].iloc[0]
            if duration >= pd.Timedelta(minutes=3):
                # Save to file
                trip.to_csv(f"{save_path}/{file_list[:-4]}-trip-{trip_counter}.csv", index=False)
                trip_counter += 1


        # for the last trip
        trip = data.loc[cut[-1]:, :]
        duration = trip['time'].iloc[-1] - trip['time'].iloc[0]
        if duration >= pd.Timedelta(minutes=3):
            trip.to_csv(f"{save_path}/{file_list[:-4]}-trip-{trip_counter}.csv", index=False)