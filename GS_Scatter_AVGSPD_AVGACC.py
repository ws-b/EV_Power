import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\ioniq5'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_distance_per_total_energy = []
all_average_speed = []
all_average_acceleration = []

for file in tqdm(file_lists):
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

    # calculate Total distance / net_discharge for each file (if net_discharge is 0, set the value to 0)
    distance_per_total_energy_km_kWh = (total_distance / 1000) / net_discharge if net_discharge != 0 else 0

    # calculate average speed (in km/h) and normalized acceleration
    avg_speed = data['emobility_spd_m_per_s'].mean() * 3.6
    avg_acceleration = (data['acceleration'] / data['acceleration'].abs().max()).mean()

    # collect all values for all files
    all_distance_per_total_energy.append(distance_per_total_energy_km_kWh)
    all_average_speed.append(avg_speed)
    all_average_acceleration.append(avg_acceleration)

# scatter plot for Total Distance / Net Discharge vs. Average Speed
plt.figure(figsize=(10, 6))
plt.scatter(all_distance_per_total_energy, all_average_speed, alpha=0.5)
plt.xlabel('Total Distance / Net Discharge (km/kWh)')
plt.ylabel('Average Speed (km/h)')
plt.title('Scatter plot of Total Distance / Net Discharge vs. Average Speed')
plt.grid(True)
plt.show()

# scatter plot for Total Distance / Net Discharge vs. Average Acceleration
plt.figure(figsize=(10, 6))
plt.scatter(all_distance_per_total_energy, all_average_acceleration, alpha=0.5)
plt.xlabel('Total Distance / Net Discharge (km/kWh)')
plt.ylabel('Normalized Average Acceleration')
plt.title('Scatter plot of Total Distance / Net Discharge vs. Normalized Average Acceleration')
plt.grid(True)
plt.show()