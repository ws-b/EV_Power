import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
def plot_model_energy_dis(file_lists, folder_path):
    all_distance_per_total_energy = []
    all_total_distances = []

    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        v = data['speed']
        v = np.array(v)

        model_power = data['Power']
        model_power = np.array(model_power)
        model_energy = model_power * t_diff / 3600 / 1000
        model_energy_cumulative = model_energy.cumsum()

        distance = v * t_diff
        total_distance = distance.cumsum()

        # calculate Total distance / Total Energy for each file (if Total Energy is 0, set the value to 0)
        distance_per_total_energy = (total_distance[-1] / 1000) / (model_energy_cumulative[-1]) if model_energy_cumulative[-1] != 0 else 0

        # collect all distance_per_total_energy values for all files
        all_distance_per_total_energy.append(distance_per_total_energy)

        # collect total distances for each file
        all_total_distances.append(total_distance[-1])

    # compute weighted mean using total distances as weights
    weighted_mean = np.dot(all_distance_per_total_energy, all_total_distances) / sum(all_total_distances)

    # plot histogram for all files
    hist_data = sns.histplot(all_distance_per_total_energy, bins='auto', color='gray', kde=False)

    # plot vertical line for weighted mean value
    plt.axvline(weighted_mean, color='red', linestyle='--')
    plt.text(weighted_mean + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Weighted Mean: {weighted_mean:.2f}', color='red', fontsize=12)

    # plot vertical line for median value
    median_value = np.median(all_distance_per_total_energy)
    plt.axvline(median_value, color='blue', linestyle='--')
    plt.text(median_value + 0.05, plt.gca().get_ylim()[1] * 0.8, f'Median: {median_value:.2f}', color='blue', fontsize=12)

    # display total number of samples at top right
    total_samples = len(all_distance_per_total_energy)
    plt.text(0.95, 0.95, f'Total Samples: {total_samples}', horizontalalignment='right',
             verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, color='black')

    # set x-axis range (from 0 to 25)
    plt.xlim(0, 25)
    plt.xlabel('Total Distance / Total Model Energy (km/kWh)')
    plt.ylabel('Number of trips')
    plt.title('Total Distance / Total Model Energy Distribution')
    plt.grid(False)
    plt.show()
def plot_bms_energy_dis(file_lists, folder_path):
    all_distance_per_total_energy = []
    all_distances = []

    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        v = data['speed']
        v = np.array(v)

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        distance = v * t_diff
        total_distance = distance.cumsum()

        # calculate Total distance / net_discharge for each file (if net_discharge is 0, set the value to 0)
        distance_per_total_energy = (total_distance[-1] / 1000) / data_energy_cumulative[-1] if data_energy_cumulative[-1] != 0 else 0

        # collect all distance_per_total_energy values for all files
        all_distance_per_total_energy.append(distance_per_total_energy)

        # collect total distance for each file
        all_distances.append(total_distance[-1])

    # compute weighted mean using total_distance as weights
    weighted_mean = np.dot(all_distance_per_total_energy, all_distances) / sum(all_distances)

    # plot histogram for all files
    hist_data = sns.histplot(all_distance_per_total_energy, bins='auto', color='gray', kde=False)

    # plot vertical line for weighted mean value
    plt.axvline(weighted_mean, color='red', linestyle='--')
    plt.text(weighted_mean + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Weighted Mean: {weighted_mean:.2f}', color='red', fontsize=12)

    # plot vertical line for median value
    median_value = np.median(all_distance_per_total_energy)
    plt.axvline(median_value, color='blue', linestyle='--')
    plt.text(median_value + 0.05, plt.gca().get_ylim()[1] * 0.8, f'Median: {median_value:.2f}', color='blue', fontsize=12)

    # display total number of samples at top right
    total_samples = len(all_distance_per_total_energy)
    plt.text(0.95, 0.95, f'Total Samples: {total_samples}', horizontalalignment='right',
             verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, color='black')

    # set x-axis range (from 0 to 25)
    plt.xlim(0, 25)
    plt.xlabel('Total Distance / Total BMS Energy (km/kWh)')
    plt.ylabel('Number of trips')
    plt.title('Total Distance / Total BMS Energy Distribution')
    plt.grid(False)
    plt.show()

def plot_fit_model_energy_dis(file_lists, folder_path):
    all_distance_per_total_energy = []
    all_distances = []

    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        v = data['speed']
        v = np.array(v)

        model_power = data['Power_fit']
        model_power = np.array(model_power)
        model_energy = model_power * t_diff / 3600 / 1000
        model_energy_cumulative = model_energy.cumsum()

        distance = v * t_diff
        total_distance = distance.cumsum()

        # calculate Total distance / Total Energy for each file (if Total Energy is 0, set the value to 0)
        distance_per_total_energy = (total_distance[-1] / 1000) / (model_energy_cumulative[-1]) if model_energy_cumulative[-1] != 0 else 0

        # collect all distance_per_total_Energy values for all files
        all_distance_per_total_energy.append(distance_per_total_energy)

        # collect total distance for each file
        all_distances.append(total_distance[-1])

    # compute weighted mean using total_distance as weights
    weighted_mean = np.dot(all_distance_per_total_energy, all_distances) / sum(all_distances)

    # plot histogram for all files
    hist_data = sns.histplot(all_distance_per_total_energy, bins='auto', color='gray', kde=False)

    # plot vertical line for weighted mean value
    plt.axvline(weighted_mean, color='red', linestyle='--')
    plt.text(weighted_mean + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Weighted Mean: {weighted_mean:.2f}', color='red', fontsize=12)

    # plot vertical line for median value
    median_value = np.median(all_distance_per_total_energy)
    plt.axvline(median_value, color='blue', linestyle='--')
    plt.text(median_value + 0.05, plt.gca().get_ylim()[1] * 0.8, f'Median: {median_value:.2f}', color='blue', fontsize=12)

    # display total number of samples at top right
    total_samples = len(all_distance_per_total_energy)
    plt.text(0.95, 0.95, f'Total Samples: {total_samples}', horizontalalignment='right',
             verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, color='black')

    # set x-axis range (from 0 to 25)
    plt.xlim(0, 25)
    plt.xlabel('Total Distance / Total Fit Model Energy (km/kWh)')
    plt.ylabel('Number of trips')
    plt.title('Total Distance / Total Fit Model Energy Distribution')
    plt.grid(False)
    plt.show()