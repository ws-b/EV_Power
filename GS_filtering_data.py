import os
import pandas as pd
import numpy as np
from tqdm import tqdm
def move_files(file_lists, folder_path, moved_path):
    # plot graphs for each file
    for file in tqdm(file_lists):
        # create file path
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # extract time, Power, CHARGE, DISCHARGE
        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        v = data['speed']
        CHARGE = data['trip_chrg_pw'].tolist()
        DISCHARGE = data['trip_dischrg_pw'].tolist()

        Energy_IV = data['Energy_IV'].tolist()

        # calculate total distance considering the sampling interval (2 seconds)
        total_distance = np.sum(v * 2)

        Energy_VI_cumulative = np.cumsum(Energy_IV).tolist()

        # calculate difference between CHARGE and DISCHARGE
        net_charge = np.array(DISCHARGE) - np.array(CHARGE)

        # convert Power data to kWh and perform cumulative calculation
        Energy_kWh = data['Energy']  # convert kW to kWh considering the 2-second time interval
        Energy_kWh_cumulative = Energy_kWh.cumsum().tolist()

        # move file if the time range is more than 5 minutes and total distance is less than 1000 and Power_kWh_cumulative is less than 0
        time_range = t.iloc[-1] - t.iloc[0]
        if time_range.total_seconds() < 300  or total_distance < 1000 or Energy_kWh_cumulative[-1] < 0 or Energy_VI_cumulative[-1] < 0:    # 5 minutes = 300 seconds
            os.replace(file_path, os.path.join(moved_path, file))

    print("Done!")