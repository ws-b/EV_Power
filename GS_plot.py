import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress
from tqdm import tqdm

def plot_power(file_lists, folder_path, Target):
    print("Plotting Power, Put Target : stacked, model, data, comparison, difference, d_altitude")
    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        data_power = np.array(data['Power_IV']) / 1000
        model_power = np.array(data['Power']) / 1000
        power_diff = data_power - model_power

        if Target == 'stacked':
            A = data['A'] / 1000
            B = data['B'] / 1000
            C = data['C'] / 1000
            D = data['D'] / 1000
            E = data['E'] / 1000

            plt.figure(figsize=(12, 6))

            plt.stackplot(t_min, A, B, C, D, E,
                          labels=['A (First)', 'B (Second)', 'C (Third)', 'D (Accel)', 'E (Aux,Idle)'],
                          colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'], edgecolor=None)
            plt.title('Power Stacked Graph Term by Term')
            plt.xlabel('Time')
            plt.ylabel('Power (W)')
            plt.legend(loc='upper left')

            plt.show()

        elif Target == 'model':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Power and Model Power (kW)')
            plt.plot(t_min, data_power, label='Data Power (kW)', color='tab:blue')


            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Model Power vs. Data Power')
            plt.tight_layout()
            plt.show()

        elif Target == 'data':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Power (kW)')
            plt.plot(t_min, data_power, label='Data Power (kW)', color='tab:blue')

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Data Power vs. Model Power')
            plt.tight_layout()
            plt.show()

        elif Target == 'comparison':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Power and Model Power (kW)')
            plt.plot(t_min, data_power, label='Data Power (kW)', color='tab:blue')
            plt.plot(t_min, model_power, label='model Power (kW)', color='tab:red')
            plt.ylim([-100, 100])

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Data Power vs.  Model Power')
            plt.tight_layout()
            plt.show()

        elif Target == 'difference':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Power - Model Power (kW)')
            plt.plot(t_min, power_diff, label='Data Power - Model Power (kW)', color='tab:blue')

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Data Power & Model Power Difference')
            plt.tight_layout()
            plt.show()

        elif Target == 'd_altitude' and 'delta altitude' in data.columns:
            # 고도 데이터
            d_altitude = np.array(data['delta altitude'])

            # 그래프 그리기
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # 첫 번째 y축 (왼쪽): 에너지 데이터
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Power (kW)')
            ax1.plot(t_min, data_power, label='Data Power (kW)', color='tab:blue')
            ax1.plot(t_min, model_power, label='Model Power (kW)', color='tab:red')
            ax1.tick_params(axis='y')

            # 두 번째 y축 (오른쪽): 고도 데이터
            ax2 = ax1.twinx()
            ax2.set_ylabel('Altitude (m)', color='tab:green')
            ax2.set_ylim([-2, 2])
            ax2.plot(t_min, d_altitude, label='Delta Altitude (m)', color='tab:green')
            ax2.tick_params(axis='y', labelcolor='tab:green')

            # 파일과 날짜 추가
            date = t.iloc[0].strftime('%Y-%m-%d')
            fig.text(0.99, 0.01, date, horizontalalignment='right', color='black', fontsize=12)
            fig.text(0.01, 0.99, 'File: ' + file, verticalalignment='top', color='black', fontsize=12)

            # 범례와 타이틀
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            plt.title('Data Power vs. Model Power and Delta Altitude')

            # 그래프 출력
            plt.tight_layout()
            plt.show()

        else:
            print("Invalid Target")
            return

def plot_energy(file_lists, folder_path, Target):
    print("Plotting Energy, Put Target : model, data, comparison, altitude, d_altitude")
    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        data_power = np.array(data['Power_IV'])
        data_energy = data_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        model_power = np.array(data['Power'])
        model_energy = model_power * t_diff / 3600 / 1000
        model_energy_cumulative = model_energy.cumsum()

        if Target == 'model':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Model Energy (kWh)')
            plt.plot(t_min, model_energy_cumulative, label='Model Energy (kWh)', color='tab:red')

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Model Energy')
            plt.tight_layout()
            plt.show()

        elif Target == 'data':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Energy (kWh)')
            plt.plot(t_min, data_energy_cumulative, label='Data Energy (kWh)', color='tab:blue')

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Data(BMS) Energy')
            plt.tight_layout()
            plt.show()

        elif Target == 'comparison':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('BMS Energy and Model Energy (kWh)')
            plt.plot(t_min, model_energy_cumulative, label='Model Energy (kWh)', color='tab:red')
            plt.plot(t_min, data_energy_cumulative, label='Data Energy (kWh)', color='tab:blue')

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Model Energy vs. BMS Energy')
            plt.tight_layout()
            plt.show()

        elif Target == 'altitude' and 'altitude' in data.columns:
            # 고도 데이터
            altitude = np.array(data['altitude'])

            # 그래프 그리기
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # 첫 번째 y축 (왼쪽): 에너지 데이터
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Energy (kWh)')
            ax1.plot(t_min, model_energy_cumulative, label='Model Energy (kWh)', color='tab:red')
            ax1.plot(t_min, data_energy_cumulative, label='Data Energy (kWh)', color='tab:blue')
            ax1.tick_params(axis='y')

            # 두 번째 y축 (오른쪽): 고도 데이터
            ax2 = ax1.twinx()
            ax2.set_ylabel('Altitude (m)', color='tab:green')  # 오른쪽 y축 레이블
            # ax2.set_ylim([0, 2000])
            ax2.plot(t_min, altitude, label='Altitude (m)', color='tab:green')
            ax2.tick_params(axis='y', labelcolor='tab:green')

            # 파일과 날짜 추가
            date = t.iloc[0].strftime('%Y-%m-%d')
            fig.text(0.99, 0.01, date, horizontalalignment='right', color='black', fontsize=12)
            fig.text(0.01, 0.99, 'File: ' + file, verticalalignment='top', color='black', fontsize=12)

            # 범례와 타이틀
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            plt.title('Model Energy vs. Data Energy and Altitude')

            # 그래프 출력
            plt.tight_layout()
            plt.show()

        else:
            print("Invalid Target")
            return


def plot_energy_scatter(file_lists, folder_path, Target):
    print('Put Target: model, fitting')
    data_energies = []
    mod_energies = []
    fitmod_energies = []
    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        data_power = np.array(data['Power_IV'])
        data_energy = data_power * t_diff / 3600 / 1000
        data_energies.append(data_energy.cumsum()[-1])

        model_power = np.array(data['Power'])
        model_energy = model_power * t_diff / 3600 / 1000
        mod_energies.append(model_energy.cumsum()[-1])

    if Target == 'model':
        # plot the graph
        fig, ax = plt.subplots(figsize=(6, 6))

        # Color map
        colors = cm.rainbow(np.linspace(0, 1, len(data_energies)))

        ax.set_xlabel('Data Energy (kWh)')
        ax.set_ylabel('Model Energy (kWh)')

        # Scatter for before fitting data
        for i in range(len(mod_energies)):
            ax.scatter(data_energies[i], mod_energies[i], color=colors[i], facecolors='none',
                       edgecolors=colors[i], label='Model Energy' if i == 0 else "")

        # Add trendline for final_energy_original and data_energies
        slope_original, intercept_original, _, _, _ = linregress(data_energies, mod_energies)
        ax.plot(np.array(data_energies), intercept_original + slope_original * np.array(data_energies),
                color='lightblue', label='Trendline')

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
        plt.title("All trip's BMS Energy vs. Model Energy over Time")
        plt.show()

    elif Target == 'fitting':
        # plot the graph
        fig, ax = plt.subplots(figsize=(6, 6))

        # Color map
        colors = cm.rainbow(np.linspace(0, 1, len(data_energies)))

        ax.set_xlabel('Data Energy (kWh)')
        ax.set_ylabel('Model Energy (kWh)')

        # Scatter for before fitting data
        for i in range(len(data_energies)):
            ax.scatter(data_energies[i], mod_energies[i], color=colors[i], facecolors='none',
                       edgecolors=colors[i], label='Before fitting' if i == 0 else "")

        # Scatter for after fitting data
        for i in range(len(data_energies)):
            ax.scatter(data_energies[i], fitmod_energies[i], color=colors[i],
                       label='After fitting' if i == 0 else "")

        # Add trendline for final_energy_original and data_energies
        slope_original, intercept_original, _, _, _ = linregress(data_energies, mod_energies)
        ax.plot(np.array(data_energies), intercept_original + slope_original * np.array(data_energies),
                color='lightblue', label='Trend (before fitting)')

        # Add trendline for after fitting
        slope, intercept, _, _, _ = linregress(data_energies, fitmod_energies)
        ax.plot(np.array(data_energies), intercept + slope * np.array(data_energies), 'b',
                label='Trend (after fitting)')

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
        plt.title("All trip's BMS Energy vs. Model Energy over Time")
        plt.show()

    else:
        print('Invalid Target')
        return

def plot_power_scatter(file_lists, folder_path):
    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        d_altitude = np.array(data['delta altitude'])
        data_power = np.array(data['Power_IV'])
        model_power = np.array(data['Power'])
        diff_power = data_power - model_power

        # Plotting for each file
        plt.figure(figsize=(10, 6))
        plt.scatter(d_altitude, diff_power, alpha=0.5)  # alpha for transparency

        plt.xlabel('Delta Altitude')
        plt.xlim(-2, 2)
        plt.ylabel('Power Difference (Data - Model)')
        plt.ylim(-40000, 40000)
        plt.title(f'Delta Altitude vs Power Difference for {file}')
        plt.grid(True)
        plt.show()

def plot_energy_dis(file_lists, folder_path, Target):
    print('Put Target: model, data, fitting')
    dis_mod_energies = []
    dis_data_energies = []
    dis_fitmod_energies = []
    total_distances = []

    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        v = data['speed']
        v = np.array(v)

        distance = v * t_diff
        total_distance = distance.cumsum()

        model_power = np.array(data['Power'])
        model_energy = model_power * t_diff / 3600 / 1000

        data_power = np.array(data['Power_IV'])
        data_energy = data_power * t_diff / 3600 / 1000

        if 'Power_fit' in data.columns:
            modfit_power = data['Power_fit']
            modfit_power = np.array(modfit_power)
            modfit_energy = modfit_power * t_diff / 3600 / 1000
            dis_fitmod_energy = ((total_distance[-1] / 1000) / (modfit_energy.cumsum()[-1])) if modfit_energy.cumsum()[
                                                                                                    -1] != 0 else 0
            dis_fitmod_energies.append(dis_fitmod_energy)
        else:
            pass

        # calculate Total distance / Total Energy for each file (if Total Energy is 0, set the value to 0)
        dis_mod_energy = ((total_distance[-1] / 1000) / (model_energy.cumsum()[-1])) if model_energy.cumsum()[
                                                                                            -1] != 0 else 0
        dis_data_energy = ((total_distance[-1] / 1000) / (data_energy.cumsum()[-1])) if data_energy.cumsum()[
                                                                                            -1] != 0 else 0

        # collect all distance_per_total_energy values for all files
        dis_mod_energies.append(dis_mod_energy)
        dis_data_energies.append(dis_data_energy)

        # collect total distances for each file
        total_distances.append(total_distance[-1])

    if Target == 'model':
        # compute weighted mean using total distances as weights
        weighted_mean = np.dot(dis_mod_energies, total_distances) / sum(total_distances)

        # plot histogram for all files
        hist_data = sns.histplot(dis_mod_energies, bins='auto', color='gray', kde=False)

        # plot vertical line for weighted mean value
        plt.axvline(weighted_mean, color='red', linestyle='--')
        plt.text(weighted_mean + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Weighted Mean: {weighted_mean:.2f}',
                 color='red', fontsize=12)

        # plot vertical line for median value
        median_value = np.median(dis_mod_energies)
        plt.axvline(median_value, color='blue', linestyle='--')
        plt.text(median_value + 0.05, plt.gca().get_ylim()[1] * 0.8, f'Median: {median_value:.2f}', color='blue',
                 fontsize=12)

        # display total number of samples at top right
        total_samples = len(dis_mod_energies)
        plt.text(0.95, 0.95, f'Total Samples: {total_samples}', horizontalalignment='right',
                 verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, color='black')

        # set x-axis range (from 0 to 25)
        plt.xlim(0, 25)
        plt.xlabel('Total Distance / Total Model Energy (km/kWh)')
        plt.ylabel('Number of trips')
        plt.title('Total Distance / Total Model Energy Distribution')
        plt.grid(False)
        plt.show()

    elif Target == 'data':
        # compute weighted mean using total distances as weights
        weighted_mean = np.dot(dis_data_energies, total_distances) / sum(total_distances)

        # plot histogram for all files
        hist_data = sns.histplot(dis_data_energies, bins='auto', color='gray', kde=False)

        # plot vertical line for weighted mean value
        plt.axvline(weighted_mean, color='red', linestyle='--')
        plt.text(weighted_mean + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Weighted Mean: {weighted_mean:.2f}',
                 color='red', fontsize=12)

        # plot vertical line for median value
        median_value = np.median(dis_data_energies)
        plt.axvline(median_value, color='blue', linestyle='--')
        plt.text(median_value + 0.05, plt.gca().get_ylim()[1] * 0.8, f'Median: {median_value:.2f}', color='blue',
                 fontsize=12)

        # display total number of samples at top right
        total_samples = len(dis_data_energies)
        plt.text(0.95, 0.95, f'Total Samples: {total_samples}', horizontalalignment='right',
                 verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, color='black')

        # set x-axis range (from 0 to 25)
        plt.xlim(0, 25)
        plt.xlabel('Total Distance / Total Model Energy (km/kWh)')
        plt.ylabel('Number of trips')
        plt.title('Total Distance / Total Model Energy Distribution')
        plt.grid(False)
        plt.show()

    elif Target == 'fitting' and 'Power_fit' in data.columns:
        # compute weighted mean using total distances as weights
        weighted_mean = np.dot(dis_fitmod_energies, total_distances) / sum(total_distances)

        # plot histogram for all files
        hist_data = sns.histplot(dis_fitmod_energies, bins='auto', color='gray', kde=False)

        # plot vertical line for weighted mean value
        plt.axvline(weighted_mean, color='red', linestyle='--')
        plt.text(weighted_mean + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Weighted Mean: {weighted_mean:.2f}',
                 color='red', fontsize=12)

        # plot vertical line for median value
        median_value = np.median(dis_fitmod_energies)
        plt.axvline(median_value, color='blue', linestyle='--')
        plt.text(median_value + 0.05, plt.gca().get_ylim()[1] * 0.8, f'Median: {median_value:.2f}', color='blue',
                 fontsize=12)

        # display total number of samples at top right
        total_samples = len(dis_fitmod_energies)
        plt.text(0.95, 0.95, f'Total Samples: {total_samples}', horizontalalignment='right',
                 verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, color='black')

        # set x-axis range (from 0 to 25)
        plt.xlim(0, 25)
        plt.xlabel('Total Distance / Total Fitted Model Energy (km/kWh)')
        plt.ylabel('Number of trips')
        plt.title('Total Distance / Total Fitted Model Energy Distribution')
        plt.grid(False)
        plt.show()

    else:
        print("Invalid Target. Please try again.")
        return