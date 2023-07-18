import os
import numpy as np
import pandas as pd
from tqdm import tqdm

win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\processed'
win_save_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\speed-acc'

folder_path = os.path.normpath(win_folder_path)
save_path = os.path.normpath(win_save_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

for file in tqdm(file_lists):
    file_path = os.path.join(folder_path, file)

    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, dtype={'device_no': str, 'measured_month': str})

    # reverse the DataFrame based on the index
    df = df[::-1]

    # convert speed unit from km/h to m/s
    df['emobility_spd_m_per_s'] = df['emobility_spd'] * 0.27778

    # calculate time and speed changes
    df['time'] = df['time'].str.strip()
    df['time'] = pd.to_datetime(df['time'], format='%y-%m-%d %H:%M:%S')
    df['time_diff'] = df['time'].diff().dt.total_seconds()
    df['emobility_spd_m_per_s'] = df['emobility_spd'] * 0.27778
    df['spd_diff'] = df['emobility_spd_m_per_s'].diff()

    # calculate acceleration
    df['acceleration'] = df['spd_diff'] / df['time_diff']

    # replace NaN values with 0 or fill with desired values
    df['acceleration'] = df['acceleration'].replace(np.nan, 0)

    # merge selected columns into a single DataFrame
    data_save = df[['time', 'emobility_spd_m_per_s', 'acceleration', 'trip_chrg_pw', 'trip_dischrg_pw', 'chrg_cable_conn', 'ext_temp', 'int_temp', 'soc', 'soh','cell_volt_list', 'pack_current', 'pack_volt','cumul_pw_chrgd', 'cumul_pw_dischrgd' ]].copy()

    # save as a CSV file
    data_save.to_csv(os.path.join(save_path,f"{df['device_no'].iloc[0].replace(' ', '')}{'-0' + df['measured_month'].iloc[0][-2:].replace(' ', '')}.csv"), index=False)
