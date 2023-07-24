import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

win_folder_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터_후처리\230710'

folder_path = os.path.normpath(win_folder_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_total_distance = []
all_total_time = []

for file in tqdm(file_lists):
    # Create the file path
    file_path = os.path.join(folder_path, file)

    # Load the data
    data = pd.read_csv(file_path)

    # Convert 'time' column to datetime
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

    # Calculate total time in seconds
    total_time = (data['time'].iloc[-1] - data['time'].iloc[0]).total_seconds()

    # Compute total distance in meters
    # Convert speed from km/h to m/s and multiply by the time interval
    data['speed_m_s'] = data['speed'] * 1000 / 3600
    data['time_diff'] = data['time'].diff().dt.total_seconds()
    total_distance = (data['speed_m_s'] * data['time_diff']).sum()

    # Save the total distance and total time for each file
    all_total_distance.append(total_distance)
    all_total_time.append(total_time)

# Convert total distance to kilometers
all_total_distance_km = [i / 1000 for i in all_total_distance]

# Histogram of total distance
plt.figure()
sns.histplot(all_total_distance_km, bins='auto', color='blue', kde=False)
plt.xlabel('Total Distance (km)')
plt.ylabel('Number of Trips')
plt.title('Total Distance Distribution')
plt.xlim(0, 100)
plt.grid(False)
plt.annotate(f'Sample size: {len(all_total_distance_km)}', xy=(0.7, 0.9), xycoords='axes fraction')
plt.show()

# Convert total time to minutes
all_total_time_min = [i / 60 for i in all_total_time]

# Histogram of total time
plt.figure()
sns.histplot([i / 60 for i in all_total_time], bins='auto', color='green', kde=False)
plt.xlabel('Total Time (min)')
plt.ylabel('Number of Trips')
plt.title('Total Time Distribution')
plt.xlim(0, 8000 / 60)  # adjust the limit to minutes
plt.grid(False)
plt.annotate(f'Sample size: {len(all_total_time_min)}', xy=(0.7, 0.9), xycoords='axes fraction')
plt.show()


# Create a scatter plot
plt.figure()
plt.scatter(all_total_time_min, all_total_distance_km)
plt.xlabel('Total Time (min)')
plt.ylabel('Total Distance (km)')
plt.title('Total Time vs Total Distance')
plt.grid(True)
plt.annotate(f'Sample size: {len(all_total_time_min)}', xy=(0.7, 0.9), xycoords='axes fraction')
plt.show()
