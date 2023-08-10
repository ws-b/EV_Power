import os
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from scipy.optimize import minimize
from tqdm import tqdm


def linear_func(v, T, a, b, c):
    return a + b * v + c * T


def objective(params, speed, temp, Power, Power_IV):
    a, b, c = params
    fitting_power = Power * linear_func(speed, temp, a, b, c)
    costs = ((fitting_power - Power_IV) ** 2).sum()
    return costs


def fit_power(file, folder_path, num_starts=10):
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)
    speed = data['speed']
    temp = data['ext_temp']
    Power = data['Power']
    Power_IV = data['Power_IV']

    best_result = None
    best_value = float('inf')

    for _ in range(num_starts):
        initial_guess = np.random.rand(3) * 10  # 초기값을 임의로 설정
        result = minimize(objective, initial_guess, args=(speed, temp, Power, Power_IV), method='BFGS')

        if result.fun < best_value:
            best_value = result.fun
            best_result = result

    a, b, c = best_result.x

    data['Power_fit'] = Power * linear_func(speed, temp, a, b, c)
    data.to_csv(file_path, index=False)

    return a, b, c


def fitting(file_lists, folder_path):
    for file in tqdm(file_lists):
        fit_power(file, folder_path)
    print("Done")
def plot_fit_model_energy_dis(file_lists, folder_path):
    all_distance_per_total_energy = []

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
    plt.xlabel('Total Distance / Total Fit Model Energy (km/kWh)')
    plt.ylabel('Number of trips')
    plt.title('Total Distance / Total Fit Model Energy Distribution')
    plt.grid(False)
    plt.show()
def plot_fit_energy_comparison(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        model_power = data['Power_fit']
        model_power = np.array(model_power)
        model_energy = model_power * t_diff / 3600 / 1000
        model_energy_cumulative = model_energy.cumsum()

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        plt.xlabel('Time (minutes)')
        plt.ylabel('BMS Energy and Fit Model Energy (kWh)')
        plt.plot(t_min, model_energy_cumulative, label='Fit Model Energy (kWh)', color='tab:blue')
        plt.plot(t_min, data_energy_cumulative, label='BMS Energy (kWh)', color='tab:red')

        # Add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
        plt.title('Fit Model Energy vs. BMS Energy')
        plt.tight_layout()
        plt.show()

def plot_fit_power_comparison(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        bms_power = data['Power_IV'] / 1000
        model_power = data['Power_fit'] / 1000

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        plt.xlabel('Time (minutes)')
        plt.ylabel('BMS Power and Fit Model Power (kW)')
        plt.plot(t_min, bms_power, label='BMS Power (kW)', color='tab:blue')
        plt.plot(t_min, model_power, label='Fit Model Power (kW)', color='tab:red')
        # plt.plot(t_min, A_power, label='v Term (kW)', color='tab:orange')
        # plt.plot(t_min, B_power, label='v^2 Term (kW)', color='tab:purple')
        # plt.plot(t_min, C_power, label='v^3 Term (kW)', color='tab:pink')
        # plt.plot(t_min, D_power, label='Acceleration Term (kW)', color='tab:green')
        # plt.plot(t_min, E_power, label='Aux/Idle Term (kW)', color='tab:brown')

        # Add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
        plt.title('Fit Model Power vs. BMS Power')
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
            color='lightblue', label='Fit prior Data')

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

def plot_contour(folder_path):
    file_path = folder_path
    data = pd.read_csv(file_path)
    speed = data['speed']
    temp = data['ext_temp']
    Power = data['Power']
    Power_IV = data['Power_IV']

    a_values = np.linspace(-10, 10, 100)
    b_values = np.linspace(-10, 10, 100)
    A, B = np.meshgrid(a_values, b_values)

    Z = np.zeros_like(A)
    c_mean = np.mean(data['Power_fit'])  # c의 평균값

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Z[i, j] = objective([A[i, j], B[i, j], c_mean], speed, temp, Power, Power_IV)

    plt.contourf(A, B, Z, 20, cmap='RdGy')
    plt.colorbar()
    plt.title("Objective Function Contour Plot")
    plt.xlabel("a value")
    plt.ylabel("b value")
    plt.show()