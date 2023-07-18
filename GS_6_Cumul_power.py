import os
import numpy as np
import pandas as pd
from tqdm import tqdm

win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\ioniq5'

folder_path = os.path.normpath(win_folder_path)

# Get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# Iterate over each file
for file in tqdm(file_lists):
    file_path = os.path.join(folder_path, file)

    # Read the data from the file into a pandas DataFrame
    data = pd.read_csv(file_path)

    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    # first_charge_value = data.loc[0, 'cumul_pw_chrgd']
    # first_discharge_value = data.loc[0, 'cumul_pw_dischrgd']
    #
    # data['Trip_charge'] = data['cumul_pw_chrgd'] - first_charge_value
    # data['Trip_discharge'] = data['cumul_pw_dischrgd'] - first_discharge_value

    data['IV'] = data['pack_volt'] * data['pack_current']

    # Calculate time difference in seconds
    t_diff = t.diff().dt.total_seconds()
    Power = data['IV'].tolist()
    # Convert lists to numpy arrays for vectorized operations
    Power = np.array(Power)
    t_diff = np.array(t_diff.fillna(0))

    # Calculate energy by multiplying power with time difference
    # Convert power from watts to kilowatts and time from seconds to hours
    Energy = Power * t_diff / 3600 / 1000

    # Convert the energy back to a list and add it to the DataFrame
    data['Energy_VI'] = Energy.tolist()

    # Overwrite the data to the same .csv file
    data.to_csv(os.path.join(folder_path, file), index=False)

print('Done')
