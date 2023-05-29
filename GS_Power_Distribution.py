import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# folder path where the files are stored
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\kona_ev\\'
mac_folder_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터 속도-가속도 처리'

folder_path = win_folder_path

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_distance_per_total_power = []
over_30_files = []

for file_list in file_lists:
    # create file path
    file_path = os.path.join(folder_path, file_list)
    data = pd.read_csv(file_path)

    # extract time, speed, Power
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    v = data['emobility_spd_m_per_s']
    Power = data['Power']

    # calculate total distance considering the sampling interval (2 seconds)
    total_distance = np.sum(v * 2)

    # total Power sum
    total_power = np.sum(Power)

    # calculate total time sum
    total_time = np.sum(t.diff().dt.total_seconds())

    # calculate Total distance / Total Power for each file (if Total Power is 0, set the value to 0)
    distance_per_total_power_km_kWh = (total_distance / 1000) / ((total_power / 1000) * (total_time / 3600)) if total_power != 0 else 0

    # collect all distance_per_total_power values for all files
    all_distance_per_total_power.append(distance_per_total_power_km_kWh)

    # if distance_per_total_power_km_kWh is over 30, add the file name to over_30_files
    if distance_per_total_power_km_kWh >= 30:
        over_30_files.append(file_list)

# plot histogram for all files
hist_data = sns.histplot(all_distance_per_total_power, bins='auto', color='gray', kde=False)

# plot vertical line for mean value
mean_value = np.mean(all_distance_per_total_power)
plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')

# display mean value
plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red', fontsize=12)

# set x-axis range (from 0 to 25)
plt.xlim(0, 25)
plt.xlabel('Total Distance / Total Power (km/kWh)')
plt.ylabel('Number of trips')
plt.title('Total Distance / Total Power Distribution')
plt.grid(False)
plt.show()

# print files with a Total Distance / Total Power ratio greater than 50
print("Files with a ratio of Total Distance / Total Power greater than 50:")
for file in over_30_files:
    print(file)
