import os
import pandas as pd
import numpy as np
from tqdm import tqdm
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

        # Calculate time difference in seconds
        Power = df['Power_IV'].tolist()

        # Convert lists to numpy arrays for vectorized operations
        Power = np.array(Power)
        t_diff = np.array(t_diff.fillna(0))

        # Calculate energy by multiplying power with time difference
        Energy = Power * t_diff / 3600 / 1000

        # Convert the energy back to a list and add it to the DataFrame
        df['Energy_IV'] = Energy.tolist()

        # merge selected columns into a single DataFrame
        data_save = df[['time', 'speed', 'acceleration', 'trip_chrg_pw', 'trip_dischrg_pw', 'pack_current', 'pack_volt',
                        'chrg_cable_conn', 'ext_temp', 'int_temp', 'soc', 'soh', 'cell_volt_list', 'Power_IV',
                        'Energy_IV']].copy()

        # save as a CSV file
        data_save.to_csv(os.path.join(save_path,
                                      f"{df['device_no'].iloc[0].replace(' ', '')}{'-0' + df['measured_month'].iloc[0][-2:].replace(' ', '')}.csv"),
                         index=False)

    print('Done')