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

        # extract time, Energy, CHARGE, DISCHARGE
        v = data['speed']
        bms_power = data['Power_IV']
        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        # calculate total distance considering the sampling interval (2 seconds)
        v = np.array(v)
        distance = v * t_diff
        total_distance = distance.cumsum()[-1]

        # convert power data to kWh and perform cumulative calculation
        model_power = data['Power']
        model_power = np.array(model_power)
        model_energy = model_power * t_diff / 3600 / 1000
        model_energy_cumulative = model_energy.cumsum()

        # move file if the time range is more than 5 minutes and total distance is less than 1000 and Power_kWh_cumulative is less than 0
        time_range = t.iloc[-1] - t.iloc[0]
        time_limit = 300
        distance_limit = 1000
        Energy_limit = 0
        if time_range.total_seconds() < time_limit or total_distance < distance_limit or model_energy_cumulative[-1] < Energy_limit or  data_energy_cumulative[-1] < Energy_limit:    # 5 minutes = 300 seconds
            os.replace(file_path, os.path.join(moved_path, file))

    print("Done!")