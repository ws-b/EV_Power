import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress
import matplotlib.cm as cm
from scipy.optimize import curve_fit

def plot_scatter_all_trip(file_lists, folder_path):
    final_energy_data = []
    final_energy = []

    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()
        final_energy_data.append(data_energy_cumulative[-1])

        model_power = data['Power']
        model_power = np.array(model_power)
        model_energy = model_power * t_diff / 3600 / 1000
        model_energy_cumulative = model_energy.cumsum()
        final_energy.append(model_energy_cumulative[-1])

    # plot the graph
    fig, ax = plt.subplots(figsize=(6, 6))  # set the size of the graph

    # Color map
    colors = cm.rainbow(np.linspace(0, 1, len(final_energy)))

    ax.set_xlabel('Model Energy (kWh)')
    ax.set_ylabel('BMS Energy (kWh)')

    for i in range(len(final_energy)):
        ax.scatter(final_energy[i], final_energy_data[i], color=colors[i])

    # Add trendline
    slope, intercept, r_value, p_value, std_err = linregress(final_energy, final_energy_data)
    ax.plot(np.array(final_energy), intercept + slope * np.array(final_energy), 'b', label='fitted line')

    # Create y=x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.title("All trip's BMS Energy vs. Model Energy over Time")
    plt.show()

def plot_scatter_tbt(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        model_power = data['Power']
        model_power = np.array(model_power)
        model_energy = model_power * t_diff / 3600 / 1000
        model_energy_cumulative = model_energy.cumsum()

        # plot the graph
        fig, ax = plt.subplots(figsize=(6, 6))  # set the size of the graph

        ax.set_xlabel('Model Energy (kWh)')
        ax.set_ylabel('BMS Energy (kWh)')
        ax.scatter(model_energy_cumulative, data_energy_cumulative, color='tab:blue')

        # Create y=x line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: ' + file, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.title('BMS Energy vs. Model Energy')
        plt.show()
def polynomial(x, a, b, c):
    return a * x**2 + b * x + c
def plot_temp_energy(file_lists, folder_path):
    all_distance_per_total_energy = []
    ext_temp_avg = []

    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # Get the average of the ext_temp column
        ext_temp_mean = data['ext_temp'].mean()
        ext_temp_avg.append(ext_temp_mean)

        v = data['speed']
        v = np.array(v)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        distance = v * t_diff
        total_distance = distance.cumsum()

        distance_per_total_energy = (total_distance[-1] / 1000) / data_energy_cumulative[-1] if data_energy_cumulative[-1] != 0 else 0
        all_distance_per_total_energy.append(distance_per_total_energy)

    # plot the graph
    fig, ax = plt.subplots(figsize=(8, 6))  # set the size of the graph

    ax.set_xlabel('Average External Temperature')
    ax.set_ylabel('BMS Mileage (km/kWh)')

    # Scatter plot
    ax.scatter(ext_temp_avg, all_distance_per_total_energy, c='b')

    # Add trendline
    slope, intercept, _, _, _ = linregress(ext_temp_avg, all_distance_per_total_energy)
    ax.plot(ext_temp_avg, intercept + slope * np.array(ext_temp_avg), 'r')

    plt.ylim(0, 15)
    plt.title("Average External Temperature vs. BMS Energy")
    plt.show()

def plot_distance_energy(file_lists, folder_path):
    all_distance_per_total_energy = []
    all_distance = []

    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        v = data['speed']
        v = np.array(v)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        distance = v * t_diff
        total_distance = distance.cumsum()

        distance_per_total_energy = (total_distance[-1] / 1000) / data_energy_cumulative[-1] if data_energy_cumulative[-1] != 0 else 0
        all_distance_per_total_energy.append(distance_per_total_energy)
        all_distance.append(total_distance[-1] / 1000)

    # plot the graph
    fig, ax = plt.subplots(figsize=(8, 6))  # set the size of the graph

    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('BMS Mileage (km/kWh)')

    # Scatter plot
    ax.scatter(all_distance, all_distance_per_total_energy, c='b')
    plt.xlim(0, 100)
    plt.ylim(3, 10)
    plt.title("Distance vs. BMS Energy")
    plt.show()

def plot_temp_energy_wh_mile(file_lists, folder_path):
    all_wh_per_mile = []
    ext_temp_avg_fahrenheit = []  # Fahrenheit temperatures

    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # Get the average of the ext_temp column and convert to Fahrenheit
        ext_temp_mean = data['ext_temp'].mean()
        ext_temp_mean_fahrenheit = (9/5) * ext_temp_mean + 32
        ext_temp_avg_fahrenheit.append(ext_temp_mean_fahrenheit)

        v = data['speed']
        v = np.array(v)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600
        data_energy_cumulative = data_energy.cumsum()

        distance = v * t_diff
        total_distance = distance.cumsum()

        distance_per_total_energy = (total_distance[-1] / 1000) / data_energy_cumulative[-1] if data_energy_cumulative[-1] != 0 else 0
        # Convert to Wh/mile
        wh_per_mile = 1 / (distance_per_total_energy * 0.621371)
        all_wh_per_mile.append(wh_per_mile)

    # plot the graph
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlabel('Average External Temperature (°F)')  # Updated label
    ax.set_ylabel('BMS Mileage (Wh/mile)')

    # Scatter plot
    ax.scatter(ext_temp_avg_fahrenheit, all_wh_per_mile, c='b')  # Use Fahrenheit temperatures

    # Polynomial curve fitting
    params, _ = curve_fit(polynomial, ext_temp_avg_fahrenheit, all_wh_per_mile)
    x_range = np.linspace(min(ext_temp_avg_fahrenheit), max(ext_temp_avg_fahrenheit), 1000)
    y_range = polynomial(x_range, *params)
    ax.plot(x_range, y_range, 'r')
    plt.xlim(-20, 120)
    plt.ylim(100, 600)
    plt.title("Average External Temperature vs. BMS Energy")
    plt.show()