import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from matplotlib.colors import Normalize
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
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

    # Color map
    colors = cm.rainbow(np.linspace(0, 1, len(all_distance_per_total_energy)))

    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('BMS Mileage (km/kWh)')

    # Scatter plot
    for i in range(len(all_distance_per_total_energy)):
        ax.scatter(all_distance[i], all_distance_per_total_energy[i], color=colors[i])
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

def plot_energy_temp_speed(file_lists, folder_path):
    all_distance_per_total_energy = []
    avg_temps = []
    avg_speeds = []

    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # 속도 변환 (m/s -> km/h)
        v = data['speed'] * 3.6
        v = np.array(v)
        avg_speed = v.mean()
        avg_speeds.append(avg_speed)

        # 평균 온도 계산
        avg_temp = data['ext_temp'].mean()
        avg_temps.append(avg_temp)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        distance = v * t_diff / 3600  # km/h로 속도를 변환했기 때문에 시간 차이도 시간 단위로 맞춰줍니다.
        total_distance = distance.cumsum()

        distance_per_total_energy = (total_distance[-1]) / data_energy_cumulative[-1] if data_energy_cumulative[
                                                                                                    -1] != 0 else 0
        all_distance_per_total_energy.append(distance_per_total_energy)

    # Color map
    colors = cm.rainbow(np.linspace(0, 1, 10))  # 10km/h마다 색깔이 바뀌므로 10개의 색상을 생성

    # 속도에 따른 색상 매핑
    speed_colors = [colors[min(int(speed // 10), 9)] for speed in avg_speeds]

    # plot the graph
    fig, ax = plt.subplots(figsize=(12, 6))  # set the size of the graph

    ax.set_xlabel('Average Temperature (°C)')
    ax.set_ylabel('BMS Mileage (km/kWh)')

    # Scatter plot
    for i in range(len(all_distance_per_total_energy)):
        ax.scatter(avg_temps[i], all_distance_per_total_energy[i], color=speed_colors[i],
                   label=f"{avg_speeds[i]:.2f} km/h")

    norm = Normalize(vmin=0, vmax=100)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.rainbow), ax=ax, label="Average Speed (km/h)")
    plt.title("Average Temperature vs. BMS Energy with Average Speed")
    # Set custom grid ticks and y-axis range
    x_ticks = np.arange(-10, 36, 1)
    y_ticks = np.arange(4.0, 12.5, 0.5)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xlim(-10, 35)
    ax.set_ylim(4.0, 12.0)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_energy_temp_speed_3d(file_lists, folder_path):
    all_distance_per_total_energy = []
    avg_temps = []
    avg_speeds = []

    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # 속도 변환 (m/s -> km/h)
        v = data['speed'] * 3.6
        v = np.array(v)
        avg_speed = v.mean()
        avg_speeds.append(avg_speed)

        # 평균 온도 계산
        avg_temp = data['ext_temp'].mean()
        avg_temps.append(avg_temp)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        distance = v * t_diff / 3600
        total_distance = distance.cumsum()

        distance_per_total_energy = (total_distance[-1]) / data_energy_cumulative[-1] if data_energy_cumulative[-1] != 0 else 0
        all_distance_per_total_energy.append(distance_per_total_energy)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Average Temperature (°C)')
    ax.set_ylabel('Average Speed (km/h)')
    ax.set_zlabel('BMS Mileage (km/kWh)')

    colors = cm.rainbow(np.linspace(0, 1, 10))
    speed_colors = [colors[min(int(speed // 10), 9)] for speed in avg_speeds]

    for i in range(len(all_distance_per_total_energy)):
        ax.scatter(avg_temps[i], avg_speeds[i], all_distance_per_total_energy[i], color=speed_colors[i])

    plt.title("3D plot of Average Temperature, Average Speed and BMS Energy")
    plt.tight_layout()
    plt.show()

def plot_energy_temp_speed_normalized(file_lists, folder_path):
    all_distance_per_total_energy = []
    avg_temps = []
    avg_speeds = []
    total_distances = []

    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # 속도 변환 (m/s -> km/h)
        v = data['speed'] * 3.6
        v = np.array(v)
        avg_speed = v.mean()
        avg_speeds.append(avg_speed)

        # 평균 온도 계산
        avg_temp = data['ext_temp'].mean()
        avg_temps.append(avg_temp)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        distance = v * t_diff / 3600
        total_distance = distance.cumsum()

        total_distances.append(total_distance[-1])

        distance_per_total_energy = (total_distance[-1]) / data_energy_cumulative[-1] if data_energy_cumulative[
                                                                                             -1] != 0 else 0
        all_distance_per_total_energy.append(distance_per_total_energy)

    avg_mileage = np.mean(all_distance_per_total_energy)

    def get_color_based_on_distance(distance):
        if distance >= 80:
            return 'yellow'
        elif distance >= 32:
            return 'orange'
        elif distance >= 8:
            return 'purple'
        else:
            return 'green'

    distance_colors = [get_color_based_on_distance(dist) for dist in total_distances]
    normalized_distances = [dist / avg_mileage for dist in all_distance_per_total_energy]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    for i in range(len(normalized_distances)):
        ax.scatter(avg_temps[i], normalized_distances[i], color=distance_colors[i])

    # Spline for smooth trendlines
    for color in ['green', 'purple', 'orange', 'yellow']:
        indices = [i for i, c in enumerate(distance_colors) if c == color]

        # Sort the temperatures for proper spline fitting
        sorted_indices = np.argsort(np.array(avg_temps)[indices])
        x_sorted = np.array(avg_temps)[indices][sorted_indices]
        y_sorted = np.array(normalized_distances)[indices][sorted_indices]

        # Create a spline fit
        spline = UnivariateSpline(x_sorted, y_sorted)

        x_range = np.linspace(min(x_sorted), max(x_sorted), 100)
        y_range = spline(x_range)
        ax.plot(x_range, y_range, color=color, alpha=0.5, linestyle='--')

    ax.set_xlabel('Average Temperature (°C)')
    ax.set_ylabel('Normalized BMS Mileage (km/kWh)')

    norm = Normalize(vmin=0, vmax=100)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.rainbow), ax=ax, label="Average Speed (km/h)")
    plt.title("Average Temperature vs. Normalized BMS Energy")
    plt.tight_layout()
    plt.show()

def plot_fit_scatter_all_trip(file_lists, folder_path):
    final_energy_data = []
    final_energy_fit = []
    final_energy_original = []

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
        final_energy_original.append(model_energy_cumulative[-1])

        model_power_fit = data['Power_fit']
        model_power_fit = np.array(model_power_fit)
        model_energy_fit = model_power_fit * t_diff / 3600 / 1000
        model_energy_fit_cumulative = model_energy_fit.cumsum()
        final_energy_fit.append(model_energy_fit_cumulative[-1])

    # plot the graph
    fig, ax = plt.subplots(figsize=(6, 6))  # set the size of the graph

    # Color map
    colors = cm.rainbow(np.linspace(0, 1, len(final_energy_fit)))

    ax.set_xlabel('Fit Model Energy (kWh)')
    ax.set_ylabel('BMS Energy (kWh)')

    for i in range(len(final_energy_fit)):
        ax.scatter(final_energy_fit[i], final_energy_data[i], color=colors[i])

    # Add trendline
    slope, intercept, r_value, p_value, std_err = linregress(final_energy_fit, final_energy_data)
    ax.plot(np.array(final_energy_fit), intercept + slope * np.array(final_energy_fit), 'b', label='BMS Data')

    # Add trendline for final_energy_original and final_energy_data
    slope_original, intercept_original, _, _, _ = linregress(final_energy_original, final_energy_fit)
    ax.plot(np.array(final_energy_original), intercept_original + slope_original * np.array(final_energy_original),
            color='lightblue', label='Data before fitting')

    # Create y=x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.legend()
    plt.title("All trip's BMS Energy vs. Fit Model Energy over Time")
    plt.show()

def plot_fit_scatter_tbt(file_lists, folder_path):
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

        model_power = data['Power_fit']
        model_power = np.array(model_power)
        model_energy = model_power * t_diff / 3600 / 1000
        model_energy_cumulative = model_energy.cumsum()

        # plot the graph
        fig, ax = plt.subplots(figsize=(6, 6))  # set the size of the graph

        ax.set_xlabel('Fit Model Energy (kWh)')
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

        plt.title('BMS Energy vs. Fit Model Energy')
        plt.show()