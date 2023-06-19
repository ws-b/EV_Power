import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# folder path where the files are stored
win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\한국에너지공과대학교_샘플데이터\kona_ev'

folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_distance_per_total_power = []

for file in file_lists:
    # create file path
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # extract time, speed, CHARGE, DISCHARGE
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    v = data['emobility_spd_m_per_s'].tolist()
    CHARGE = data['trip_chrg_pw'].tolist()
    DISCHARGE = data['trip_dischrg_pw'].tolist()

    # calculate total distance considering the sampling interval (2 seconds)
    total_distance = np.sum(v * 2)

    # calculate net_discharge by subtracting CHARGE sum from DISCHARGE sum
    net_discharge = 0
    for i in range(len(DISCHARGE) - 1, -1, -1):
        net_discharge = DISCHARGE[i] - CHARGE[i]
        if net_discharge != 0:
            break
-
    # calculate Total distance / net_discharge for each file (if net_discharge is 0, set the value to 0)
    distance_per_total_power_km_kWh = (total_distance / 1000) / net_discharge if net_discharge != 0 else 0

    # collect all distance_per_total_power values for all files
    all_distance_per_total_power.append(distance_per_total_power_km_kWh)

# plot histogram for all files
hist_data = sns.histplot(all_distance_per_total_power, bins='auto', color='gray', kde=False)

# plot vertical line for mean value
mean_value = np.mean(all_distance_per_total_power)
plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')

# display mean value
plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red', fontsize=12)

# set x-axis range (from 0 to 25)
plt.xlim(0, 25)
plt.xlabel('Total Distance / Net Discharge (km/kWh)')
plt.ylabel('Number of trips')
plt.title('Total Distance / Net Discharge Distribution')
plt.grid(False)
plt.show()

# generate a list of files where Total Distance / Net Discharge depending on value
low_distance_power_files = [file for file, value in zip(file_lists, all_distance_per_total_power) if value <= 1]
high_distance_power_files = [file for file, value in zip(file_lists, all_distance_per_total_power) if value >= 15]
# output
for file in low_distance_power_files:
    print(file)

for file in high_distance_power_files:
    print(file)