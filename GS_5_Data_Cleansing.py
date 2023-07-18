import os
import pandas as pd
import numpy as np
from tqdm import tqdm

win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\ioniq5'


folder_path = os.path.normpath(win_folder_path)
moved_path = os.path.join(folder_path, 'moved')

if not os.path.exists(moved_path):
    os.makedirs(moved_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# plot graphs for each file
for file in tqdm(file_lists):
    # create file path
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # extract time, Power, CHARGE, DISCHARGE
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    v = data['emobility_spd_m_per_s']
    CHARGE = data['trip_chrg_pw'].tolist()
    DISCHARGE = data['trip_dischrg_pw'].tolist()

    # calculate difference between CHARGE and DISCHARGE
    net_charge = np.array(DISCHARGE) - np.array(CHARGE)

    # convert Power data to kWh and perform cumulative calculation
    Energy_kWh = data['Energy']  # convert kW to kWh considering the 2-second time interval
    Energy_kWh_cumulative = Energy_kWh.cumsum().tolist()

    # calculate total distance considering the sampling interval (2 seconds)
    total_distance = np.sum(v * 2)

    # move file if the time range is more than 5 minutes and total distance is less than 1000 and Power_kWh_cumulative is less than 0
    time_range = t.iloc[-1] - t.iloc[0]
    if time_range.total_seconds() < 300  or total_distance < 1000 or Energy_kWh_cumulative[-1] < 0:    # 5 minutes = 300 seconds
        os.replace(file_path, os.path.join(moved_path, file))

print("Done!")
