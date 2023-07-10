import os
import pandas as pd
import numpy as np
from tqdm import tqdm
win_folder_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터_후처리\230710'
folder_path = os.path.normpath(win_folder_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# Open each CSV file
for file in tqdm(file_lists):
    file_path = os.path.join(folder_path, file)

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)

    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    v = data['speed']
    Energy_kWh = data['Energy']
    Energy_kWh_cumulative = Energy_kWh.cumsum()

    # calculate total distance considering the sampling interval (2 seconds)
    total_distance = np.sum(v * 2)

    # only plot the graph if the time range is more than 5 minutes
    time_range = t.iloc[-1] - t.iloc[0]
    if time_range.total_seconds() < 300 or total_distance < 1000 or Energy_kWh_cumulative.iloc[-1] < 1.0:    # 5 minutes = 300 seconds , 1000 m = 1 km
        os.remove(file_path)  # delete the file

print("Done!")