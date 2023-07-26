import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm

def plot_model_energy_dis(file_lists, folder_path):
    all_distance_per_total_energy = []

    for file in tqdm(file_lists):
        # create file path
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # extract time, speed, Energy
        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        v = data['emobility_spd_m_per_s']
        Energy = data['Energy']

        # calculate total distance considering the sampling interval (2 seconds)
        total_distance = np.sum(v * 2)

        # total Energy sum
        total_energy = np.sum(Energy)

        # calculate total time sum
        total_time = np.sum(t.diff().dt.total_seconds())

        # calculate Total distance / Total Energy for each file (if Total Energy is 0, set the value to 0)
        distance_per_total_energy_km_kWh = (total_distance / 1000) / (total_energy) if total_energy != 0 else 0

        # collect all distance_per_total_Energy values for all files
        all_distance_per_total_energy.append(distance_per_total_energy_km_kWh)

    # plot histogram for all files
    hist_data = sns.histplot(all_distance_per_total_energy, bins='auto', color='gray', kde=False)

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
    plt.xlabel('Total Distance / Total Energy (km/kWh)')
    plt.ylabel('Number of trips')
    plt.title('Total Distance / Total Energy Distribution')
    plt.grid(False)
    plt.show()
def plot_bms_energy_dis(file_lists, folder_path):
    all_distance_per_total_energy = []

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

        # collect all distance_per_total_energy values for all files
        all_distance_per_total_energy.append(distance_per_total_energy_km_kWh)

    # plot histogram for all files
    hist_data = sns.histplot(all_distance_per_total_energy, bins='auto', color='gray', kde=False)

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
    plt.xlabel('Total Distance / Net Discharge (km/kWh)')
    plt.ylabel('Number of trips')
    plt.title('Total Distance / Net Discharge Distribution')
    plt.grid(False)
    plt.show()