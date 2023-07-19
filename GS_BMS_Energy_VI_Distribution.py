import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm

win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\ioniq5'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_distance_per_total_energy = []
file_lists_with_negative_values = []

for file in tqdm(file_lists):
    # create file path
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # extract time, speed, CHARGE, DISCHARGE
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    v = data['emobility_spd_m_per_s'].tolist()
    Energy_VI = data['Energy_VI'].tolist()

    # calculate total distance considering the sampling interval (2 seconds)
    total_distance = np.sum(v * 2)

    Energy_VI_cumulative = np.cumsum(Energy_VI).tolist()
    total_energy = Energy_VI_cumulative[-1]

    # calculate Total distance / net_discharge for each file (if net_discharge is 0, set the value to 0)
    distance_per_total_energy_km_kWh = (total_distance / 1000) / total_energy

    # collect all distance_per_total_energy values for all files
    all_distance_per_total_energy.append(distance_per_total_energy_km_kWh)

    # check if the value is negative
    if distance_per_total_energy_km_kWh < 0:
        file_lists_with_negative_values.append(file)

# plot histogram for all files
hist_data = sns.histplot(all_distance_per_total_energy, bins=50, color='gray', kde=False)

# plot vertical line for mean value
mean_value = np.mean(all_distance_per_total_energy)
plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')

# display mean value
plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red', fontsize=12)

# display total number of samples
total_samples = len(all_distance_per_total_energy)
plt.text(0.95, 0.95, f'Total Samples: {total_samples}', horizontalalignment='right',
         verticalalignment='top', transform=plt.gca().transAxes)

# set x-axis range (from 0 to 25)
plt.xlim(0, 25)
plt.xlabel('Total Distance / BMS Energy (km/kWh)')
plt.ylabel('Number of trips')
plt.title('Total Distance / Net Discharge Distribution')
plt.grid(False)
plt.show()