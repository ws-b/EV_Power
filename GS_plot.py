import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.stats import linregress
from tqdm import tqdm

def plot_power(file_lists, folder_path, selected_car, Target):
    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # Device Number 및 Trip Number 추출
        parts = file_path.split(os.sep)
        file_name = parts[-1]
        name_parts = file_name.split('_')
        trip_info = (name_parts[2] if 'altitude' in name_parts else name_parts[1]).split('.')[0]

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        data_power = np.array(data['Power_IV']) / 1000
        model_power = np.array(data['Power']) / 1000
        power_diff = data_power - model_power

        if 'Predicted_Power' in data.columns:
            predicted_power = np.array(data['Predicted_Power']) / 1000
        
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
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
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
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
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
            plt.plot(t_min, predicted_power, label='Predicted Power (kW)', color='tab:green')
            plt.ylim([-100, 100])

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
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
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
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
            fig.text(0.01, 0.99, f'{selected_car}: ' + trip_info, verticalalignment='top', color='black', fontsize=12)

            # 범례와 타이틀
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            plt.title('Data Power vs. Model Power and Delta Altitude')

            # 그래프 출력
            plt.tight_layout()
            plt.show()

        else:
            print("Invalid Target")
            return

def plot_energy(file_lists, folder_path, selected_car, Target):
    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # Device Number 및 Trip Number 추출
        parts = file_path.split(os.sep)
        file_name = parts[-1]
        name_parts = file_name.split('_')
        trip_info = (name_parts[2] if 'altitude' in name_parts else name_parts[1]).split('.')[0]

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        data_power = np.array(data['Power_IV'])
        data_energy = data_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()
        
        if 'Power' in data.columns:
            model_power = np.array(data['Power'])
            model_energy = model_power * t_diff / 3600 / 1000
            model_energy_cumulative = model_energy.cumsum()

        if 'Predicted_Power' in data.columns:
            predicted_power = np.array(data['Predicted_Power'])
            predicted_energy = predicted_power * t_diff / 3600 / 1000
            predicted_energy_cumulative = predicted_energy.cumsum()

        else:
            pass

        if Target == 'model':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Model Energy (kWh)')
            plt.plot(t_min, model_energy_cumulative, label='Model Energy (kWh)', color='tab:red')

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.96))
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
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.96))
            plt.title('Data(BMS) Energy')
            plt.tight_layout()
            plt.show()

        elif Target == 'learning':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Trained Model Energy (kWh)')
            plt.plot(t_min, predicted_energy_cumulative, label='Trained Model Energy (kWh)', color='tab:blue')

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.96))
            plt.title('Train Model Energy')
            plt.tight_layout()
            plt.show()


        elif Target == 'comparison':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('BMS Energy and Model Energy (kWh)')
            plt.plot(t_min, model_energy_cumulative, label='Model Energy (kWh)', color='tab:red')
            plt.plot(t_min, data_energy_cumulative, label='Data Energy (kWh)', color='tab:blue')
            plt.plot(t_min, predicted_energy_cumulative, label='Trained Model Energy (kWh)', color='tab:green')

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.96))
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
            fig.text(0.01, 0.99, f'{selected_car}: '+ trip_info, verticalalignment='top', color='black', fontsize=12)

            # 범례와 타이틀
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            plt.title('Model Energy vs. Data Energy and Altitude')

            # 그래프 출력
            plt.tight_layout()
            plt.show()

        else:
            print("Invalid Target")
            return

def calculate_rmse(true_values, predicted_values):
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    mse = np.mean((true_values - predicted_values) ** 2)
    rmse = np.sqrt(mse)
    relative_rmse = rmse / np.mean(true_values)
    return rmse, relative_rmse

def plot_energy_scatter(file_lists, selected_car, target):
    data_energies = []
    mod_energies = []
    predicted_energies = []

    all_data_energies = []
    all_mod_energies = []
    all_predicted_energies = []

    # 전체 데이터에 대한 에너지 계산
    for file in tqdm(file_lists):
        data = pd.read_csv(file)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        data_power = np.array(data['Power_IV'])
        data_energy = data_power * t_diff / 3600 / 1000
        all_data_energies.append(data_energy.cumsum()[-1])

        if 'Power' in data.columns:
            model_power = np.array(data['Power'])
            model_energy = model_power * t_diff / 3600 / 1000
            all_mod_energies.append(model_energy.cumsum()[-1])

        if 'Predicted_Power' in data.columns:
            predicted_power = np.array(data['Predicted_Power'])
            predicted_energy = predicted_power * t_diff / 3600 / 1000
            all_predicted_energies.append(predicted_energy.cumsum()[-1])

    # 랜덤으로 1000개의 파일을 선택하여 산점도 그리기
    sample_size = min(1000, len(file_lists))
    sampled_files = random.sample(file_lists, sample_size)

    for file in tqdm(sampled_files):
        data = pd.read_csv(file)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        data_power = np.array(data['Power_IV'])
        data_energy = data_power * t_diff / 3600 / 1000
        data_energies.append(data_energy.cumsum()[-1])

        if 'Power' in data.columns:
            model_power = np.array(data['Power'])
            model_energy = model_power * t_diff / 3600 / 1000
            mod_energies.append(model_energy.cumsum()[-1])

        if 'Predicted_Power' in data.columns:
            predicted_power = np.array(data['Predicted_Power'])
            predicted_energy = predicted_power * t_diff / 3600 / 1000
            predicted_energies.append(predicted_energy.cumsum()[-1])

    if target == 'model':
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = cm.rainbow(np.linspace(0, 1, len(data_energies)))

        ax.set_xlabel('Data Energy (kWh)')
        ax.set_ylabel('Model Energy (kWh)')

        for i in range(len(mod_energies)):
            ax.scatter(data_energies[i], mod_energies[i], color=colors[i], facecolors='none',
                       edgecolors=colors[i], label='Model Energy' if i == 0 else "")

        # 전체 데이터셋을 사용하여 45도 기준선 계산
        slope_original, intercept_original, _, _, _ = linregress(all_data_energies, all_mod_energies)
        ax.plot(np.array(data_energies), intercept_original + slope_original * np.array(data_energies),
                color='lightblue')

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        # RMSE 계산 및 플롯에 표기
        rmse, relative_rmse = calculate_rmse(all_data_energies, all_mod_energies)
        plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}kWh\nRelative RMSE: {relative_rmse:.2%}',
                 transform=ax.transAxes, fontsize=12, verticalalignment='top')

        plt.legend()
        plt.title(f"{selected_car} : BMS Energy vs. Model Energy")
        plt.show()

    elif target == 'learning':
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = cm.rainbow(np.linspace(0, 1, len(data_energies)))

        ax.set_xlabel('Data Energy (kWh)')
        ax.set_ylabel('Predicted Energy (kWh)')

        for i in range(len(data_energies)):
            ax.scatter(data_energies[i], mod_energies[i], color=colors[i], facecolors='none',
                       edgecolors=colors[i], label='Before learning' if i == 0 else "")

        for i in range(len(data_energies)):
            ax.scatter(data_energies[i], predicted_energies[i], color=colors[i],
                       label='After learning' if i == 0 else "")

        slope_original, intercept_original, _, _, _ = linregress(all_data_energies, all_mod_energies)
        ax.plot(np.array(data_energies), intercept_original + slope_original * np.array(data_energies),
                color='lightblue')

        slope, intercept, _, _, _ = linregress(all_data_energies, all_predicted_energies)
        ax.plot(np.array(data_energies), intercept + slope * np.array(data_energies), 'b')

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        # RMSE & RRMSE
        rmse_before, relative_rmse_before = calculate_rmse(all_data_energies, all_mod_energies)
        rmse_after, relative_rmse_after = calculate_rmse(all_data_energies, all_predicted_energies)
        plt.text(0.6, 0.15, f'RMSE (Before): {rmse_before:.2f}kWh\nRRMSE (Before): {relative_rmse_before:.2%}\nRMSE (After): {rmse_after:.2f}kWh\nRRMSE (After): {relative_rmse_after:.2%}',
                 transform=ax.transAxes, fontsize=10, verticalalignment='top')

        plt.legend()
        plt.title(f"{selected_car} : BMS Energy vs. Trained Model Energy")
        plt.show()

    else:
        print('Invalid Target')
        return

def plot_driver_energy_scatter(file_lists_dict, selected_car):
    data_energies = {}
    mod_energies = {}
    predicted_energies = {}

    colors = cm.rainbow(np.linspace(0, 1, len(file_lists_dict)))
    color_map = {}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel('Data Energy (kWh)')
    ax.set_ylabel('Predicted Energy (kWh)')

    for i, (id, files) in enumerate(file_lists_dict.items()):
        data_energies[id] = []
        mod_energies[id] = []
        predicted_energies[id] = []
        color_map[id] = colors[i]

        for file in tqdm(files, desc=f'Processing {id}'):
            data = pd.read_csv(file)

            t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            t_diff = t.diff().dt.total_seconds().fillna(0)
            t_diff = np.array(t_diff.fillna(0))

            data_power = np.array(data['Power_IV'])
            data_energy = data_power * t_diff / 3600 / 1000
            data_energies[id].append(data_energy.cumsum()[-1])

            if 'Power' in data.columns:
                model_power = np.array(data['Power'])
                model_energy = model_power * t_diff / 3600 / 1000
                mod_energies[id].append(model_energy.cumsum()[-1])

            if 'Predicted_Power' in data.columns:
                predicted_power = np.array(data['Predicted_Power'])
                predicted_energy = predicted_power * t_diff / 3600 / 1000
                predicted_energies[id].append(predicted_energy.cumsum()[-1])

        ax.scatter(data_energies[id], predicted_energies[id], facecolors='none', edgecolors=color_map[id],
                   label=f'{id} After learning')

        slope, intercept, _, _, _ = linregress(data_energies[id], predicted_energies[id])
        ax.plot(np.array(data_energies[id]), intercept + slope * np.array(data_energies[id]), color=color_map[id])

        # Calculate RMSE & RRMSE for each id
        rmse_before, relative_rmse_before = calculate_rmse(data_energies[id], mod_energies[id])
        rmse_after, relative_rmse_after = calculate_rmse(data_energies[id], predicted_energies[id])
        ax.text(0.05, 0.95 - i * 0.1, f'{id}\nRMSE (Before): {rmse_before:.2f}kWh\nRRMSE (Before): {relative_rmse_before:.2%}\nRMSE (After): {rmse_after:.2f}kWh\nRRMSE (After): {relative_rmse_after:.2%}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top', color=color_map[id])

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.legend(loc='lower right')
    plt.title(f"{selected_car} : Drivers Energy Comparison\n BMS Energy vs. Trained Model Energy over Time")

    plt.show()

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


def plot_energy_dis(file_lists, folder_path, selected_car, Target):
    dis_mod_energies = []
    dis_data_energies = []
    dis_predicted_energies = []
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

        if 'Power' in data.columns:
            model_power = np.array(data['Power'])
            model_energy = model_power * t_diff / 3600 / 1000
        else:
            model_energy = np.zeros_like(t_diff)

        data_power = np.array(data['Power_IV'])
        data_energy = data_power * t_diff / 3600 / 1000

        if 'Predicted_Power' in data.columns:
            predicted_power = data['Predicted_Power']
            predicted_power = np.array(predicted_power)
            predicted_energy = predicted_power * t_diff / 3600 / 1000
            dis_predicted_energy = ((total_distance[-1] / 1000) / (predicted_energy.cumsum()[-1])) if \
            predicted_energy.cumsum()[-1] != 0 else 0
            dis_predicted_energies.append(dis_predicted_energy)

        # calculate Total distance / Total Energy for each file (if Total Energy is 0, set the value to 0)
        dis_mod_energy = ((total_distance[-1] / 1000) / (model_energy.cumsum()[-1])) if model_energy.cumsum()[
                                                                                            -1] != 0 else 0
        dis_mod_energies.append(dis_mod_energy)

        dis_data_energy = ((total_distance[-1] / 1000) / (data_energy.cumsum()[-1])) if data_energy.cumsum()[
                                                                                            -1] != 0 else 0
        dis_data_energies.append(dis_data_energy)

        # collect total distances for each file
        total_distances.append(total_distance[-1])

    if Target == 'model':
        # compute mean value
        mean_value = np.mean(dis_mod_energies)

        # plot histogram for all files
        hist_data = sns.histplot(dis_mod_energies, bins='auto', color='gray', kde=False)

        # plot vertical line for mean value
        plt.axvline(mean_value, color='red', linestyle='--')
        plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}',
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
        plt.title(f"{selected_car} : Total Distance / Total Model Energy Distribution")
        plt.grid(False)
        plt.show()

    elif Target == 'data':
        # compute mean value
        mean_value = np.mean(dis_data_energies)

        # plot histogram for all files
        hist_data = sns.histplot(dis_data_energies, bins='auto', color='gray', kde=False)

        # plot vertical line for mean value
        plt.axvline(mean_value, color='red', linestyle='--')
        plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}',
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
        plt.xlabel('Total Distance / Total Data Energy (km/kWh)')
        plt.ylabel('Number of trips')
        plt.title(f"{selected_car} : Total Distance / Total Data Energy Distribution")
        plt.grid(False)
        plt.show()

    elif Target == 'learning' and 'Predicted_Power' in data.columns:
        # compute mean value
        mean_value = np.mean(dis_predicted_energies)
        # plot histogram for all files
        hist_data = sns.histplot(dis_predicted_energies, bins='auto', color='gray', kde=False)

        # plot vertical line for mean value
        plt.axvline(mean_value, color='red', linestyle='--')
        plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}',
                 color='red', fontsize=12)

        # plot vertical line for median value
        median_value = np.median(dis_predicted_energies)
        plt.axvline(median_value, color='blue', linestyle='--')
        plt.text(median_value + 0.05, plt.gca().get_ylim()[1] * 0.8, f'Median: {median_value:.2f}', color='blue',
                 fontsize=12)

        # display total number of samples at top right
        total_samples = len(dis_predicted_energies)
        plt.text(0.95, 0.95, f'Total Samples: {total_samples}', horizontalalignment='right',
                 verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, color='black')

        # set x-axis range (from 0 to 25)
        plt.xlim(0, 25)
        plt.xlabel('Total Distance / Total Trained Model Energy (km/kWh)')
        plt.ylabel('Number of trips')
        plt.title(f"{selected_car} : Total Distance / Total Trained Model Energy Distribution")
        plt.grid(False)
        plt.show()

    else:
        print("Invalid Target. Please try again.")
        return
def plot_3d(X, y_true, y_pred, fold_num, vehicle, scaler, num_grids=400, samples_per_grid=30, output_file=None):
    if X.shape[1] != 2:
        print("Error: X should have 2 columns.")
        return

    # 역변환하여 원래 범위로 변환
    X_orig = scaler.inverse_transform(X)

    # Speed를 km/h로 변환
    X_orig[:, 0] *= 3.6

    # 그리드 크기 계산
    num_grid_sqrt = int(np.sqrt(num_grids))
    grid_size_x = (X_orig[:, 0].max() - X_orig[:, 0].min()) / num_grid_sqrt
    grid_size_y = (X_orig[:, 1].max() - X_orig[:, 1].min()) / num_grid_sqrt

    samples = []

    for i in range(num_grid_sqrt):
        for j in range(num_grid_sqrt):
            x_min = X_orig[:, 0].min() + i * grid_size_x
            x_max = x_min + grid_size_x
            y_min = X_orig[:, 1].min() + j * grid_size_y
            y_max = y_min + grid_size_y

            # 해당 그리드 내의 데이터 인덱스 선택
            grid_indices = np.where(
                (X_orig[:, 0] >= x_min) & (X_orig[:, 0] < x_max) &
                (X_orig[:, 1] >= y_min) & (X_orig[:, 1] < y_max)
            )[0]

            if len(grid_indices) > 0:
                sample_size = min(samples_per_grid, len(grid_indices))
                sample_indices = np.random.choice(grid_indices, sample_size, replace=False)
                samples.append(sample_indices)

    samples = np.concatenate(samples)
    X_sampled = X_orig[samples]
    y_true_sampled = y_true[samples]
    y_pred_sampled = y_pred[samples]

    trace1 = go.Scatter3d(
        x=X_sampled[:, 0], y=X_sampled[:, 1], z=y_true_sampled,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.8),
        name='Actual Residual'
    )
    trace2 = go.Scatter3d(
        x=X_sampled[:, 0], y=X_sampled[:, 1], z=y_pred_sampled,
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.8),
        name='Predicted Residual'
    )

    grid_x, grid_y = np.linspace(X_orig[:, 0].min(), X_orig[:, 0].max(), 100), np.linspace(X_orig[:, 1].min(), X_orig[:, 1].max(), 100)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata((X_orig[:, 0], X_orig[:, 1]), y_pred, (grid_x, grid_y), method='linear')

    surface_trace = go.Surface(
        x=grid_x,
        y=grid_y,
        z=grid_z,
        colorscale='Viridis',
        name='Predicted Residual Surface',
        opacity=0.7
    )

    data = [trace1, trace2, surface_trace]
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='Speed (km/h)'),
            yaxis=dict(title='Acceleration (m/s²)'),
            zaxis=dict(title='Residual'),
        ),
        title=f'3D Plot of Actual vs. Predicted Residuals (Fold {fold_num}, Vehicle: {vehicle})'
    )
    fig = go.Figure(data=data, layout=layout)
    if output_file:
        fig.write_html(output_file)
    else:
        fig.show()

def plot_contour(X, y_pred, scaler, selected_car, num_grids=400, output_file=None):
    if X.shape[1] != 2:
        raise ValueError("Error: X should have 2 columns.")

    # Inverse transform to original scale
    X_orig = scaler.inverse_transform(X)

    # Convert speed to km/h
    X_orig[:, 0] *= 3.6

    # Create grid
    grid_x = np.linspace(X_orig[:, 0].min(), X_orig[:, 0].max(), num_grids)
    grid_y = np.linspace(X_orig[:, 1].min(), X_orig[:, 1].max(), num_grids)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata((X_orig[:, 0], X_orig[:, 1]), y_pred, (grid_x, grid_y), method='linear')

    # Contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title(f'{selected_car} : Contour Plot of Predicted Residuals')

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def plot_contour2(file_lists, selected_car, num_grids=400):
    all_data = []

    for file in file_lists:
        try:
            data = pd.read_csv(file)
            all_data.append(data)
        except FileNotFoundError:
            print(f"Error: File {file} not found.")
            return

    # Merge data
    merged_data = pd.concat(all_data, ignore_index=True)

    X = merged_data['speed'] * 3.6
    Y = merged_data['acceleration']
    Residual = merged_data['Power'] - merged_data['Power_IV']

    # Handle missing values
    mask = ~np.isnan(X) & ~np.isnan(Y) & ~np.isnan(Residual)
    X = X[mask]
    Y = Y[mask]
    Residual = Residual[mask]

    # Create grid
    grid_x = np.linspace(X.min(), X.max(), num_grids)
    grid_y = np.linspace(Y.min(), Y.max(), num_grids)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata((X, Y), Residual, (grid_x, grid_y), method='linear')

    # Contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title(f'{selected_car} : Contour Plot of Residuals')

    plt.show()

def plot_2d_histogram(sample_files_dict, selected_car, Target = 'data'):
    if isinstance(sample_files_dict, dict):
        for id, files in tqdm(sample_files_dict.items()):
            dis_data_energies = []
            total_distances = []
            average_speeds = []

            for file in files:
                data = pd.read_csv(file)

                t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
                t_diff = t.diff().dt.total_seconds().fillna(0)
                t_diff = np.array(t_diff)

                v = data['speed']
                v = np.array(v)

                distance = v * t_diff
                total_distance = distance.cumsum()[-1]

                data_power = np.array(data['Power_IV'])
                data_energy = data_power * t_diff / 3600 / 1000

                total_energy = data_energy.cumsum()[-1]
                dis_data_energy = ((total_distance / 1000) / total_energy) if total_energy != 0 else 0

                total_distances.append(total_distance / 1000)  # convert to km
                dis_data_energies.append(dis_data_energy)
                average_speed = np.mean(v) * 3.6  # converting m/s to km/h
                average_speeds.append(average_speed)

            # Create bins
            x_bins = np.linspace(min(total_distances), max(total_distances), 20)
            y_bins = np.linspace(min(average_speeds), max(average_speeds), 20)

            heatmap, _, _ = np.histogram2d(total_distances, average_speeds, bins=[x_bins, y_bins], weights=dis_data_energies)
            counts, _, _ = np.histogram2d(total_distances, average_speeds, bins=[x_bins, y_bins])

            # Avoid division by zero
            average_heatmap = np.divide(heatmap, counts, where=counts != 0)

            # Mask the zero values
            average_heatmap = np.ma.masked_where(counts == 0, average_heatmap)
            etamin = 2
            etamax = 11
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(x_bins, y_bins, average_heatmap.T, shading='auto', cmap='coolwarm', vmin=etamin, vmax=etamax)
            cb = plt.colorbar(label='Average Energy Efficiency (km/kWh)')

            plt.xlabel('Trip Distance (km)')
            plt.ylabel('Average Speed (km/h)')
            plt.title(f"{selected_car} ({id}) : Trip Distance vs. Average Speed with Energy Efficiency, {len(files)} files")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.show()

    elif isinstance(sample_files_dict, list):
        dis_data_energies = []
        dis_mod_energies = []
        dis_predicted_energies = []
        total_distances = []
        average_speeds = []

        for file in tqdm(sample_files_dict):
            data = pd.read_csv(file)

            t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            t_diff = t.diff().dt.total_seconds().fillna(0)
            t_diff = np.array(t_diff)

            v = data['speed']
            v = np.array(v)

            distance = v * t_diff
            total_distance = distance.cumsum()[-1]

            if 'Power' in data.columns:
                model_power = np.array(data['Power'])
                model_energy = model_power * t_diff / 3600 / 1000
            else:
                model_energy = np.zeros_like(t_diff)

            data_power = np.array(data['Power_IV'])
            data_energy = data_power * t_diff / 3600 / 1000

            if 'Predicted_Power' in data.columns:
                predicted_power = data['Predicted_Power']
                predicted_power = np.array(predicted_power)
                predicted_energy = predicted_power * t_diff / 3600 / 1000
                dis_predicted_energy = ((total_distance / 1000) / (predicted_energy.cumsum()[-1])) if \
                    predicted_energy.cumsum()[-1] != 0 else 0
                dis_predicted_energies.append(dis_predicted_energy)

            # calculate Total distance / Total Energy for each file (if Total Energy is 0, set the value to 0)
            dis_mod_energy = ((total_distance / 1000) / (model_energy.cumsum()[-1])) if model_energy.cumsum()[
                                                                                                -1] != 0 else 0
            dis_mod_energies.append(dis_mod_energy)

            dis_data_energy = ((total_distance / 1000) / (data_energy.cumsum()[-1])) if data_energy.cumsum()[
                                                                                                -1] != 0 else 0
            dis_data_energies.append(dis_data_energy)

            total_distances.append(total_distance / 1000)

            average_speed = np.mean(v) * 3.6  # converting m/s to km/h
            average_speeds.append(average_speed)
        if Target == 'model' :
            # Create bins
            x_bins = np.linspace(min(total_distances), max(total_distances), 20)
            y_bins = np.linspace(min(average_speeds), max(average_speeds), 20)

            heatmap, _, _ = np.histogram2d(total_distances, average_speeds, bins=[x_bins, y_bins], weights=dis_mod_energies)
            counts, _, _ = np.histogram2d(total_distances, average_speeds, bins=[x_bins, y_bins])

            # Avoid division by zero
            average_heatmap = np.divide(heatmap, counts, where=counts != 0)

            # Mask the zero values
            average_heatmap = np.ma.masked_where(counts == 0, average_heatmap)

            etamin = 2
            etamax = 11
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(x_bins, y_bins, average_heatmap.T, shading='auto', cmap='coolwarm', vmin=etamin, vmax=etamax)
            cb = plt.colorbar(label='Average Model Energy Efficiency (km/kWh)')

            plt.xlabel('Trip Distance (km)')
            plt.ylabel('Average Speed (km/h)')
            plt.title(f"{selected_car} : Trip Distance vs. Average Speed with Energy Efficiency, {len(sample_files_dict)} files")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.show()
        elif Target == 'data' :
            # Create bins
            x_bins = np.linspace(min(total_distances), max(total_distances), 20)
            y_bins = np.linspace(min(average_speeds), max(average_speeds), 20)

            heatmap, _, _ = np.histogram2d(total_distances, average_speeds, bins=[x_bins, y_bins],
                                           weights=dis_data_energies)
            counts, _, _ = np.histogram2d(total_distances, average_speeds, bins=[x_bins, y_bins])

            # Avoid division by zero
            average_heatmap = np.divide(heatmap, counts, where=counts != 0)

            # Mask the zero values
            average_heatmap = np.ma.masked_where(counts == 0, average_heatmap)

            etamin = 2
            etamax = 11
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(x_bins, y_bins, average_heatmap.T, shading='auto', cmap='coolwarm', vmin=etamin, vmax=etamax)
            cb = plt.colorbar(label='Average Data Energy Efficiency (km/kWh)')

            plt.xlabel('Trip Distance (km)')
            plt.ylabel('Average Speed (km/h)')
            plt.title(
                f"{selected_car} : Trip Distance vs. Average Speed with Energy Efficiency, {len(sample_files_dict)} files")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.show()
        elif Target == 'learning':
            # Create bins
            x_bins = np.linspace(min(total_distances), max(total_distances), 20)
            y_bins = np.linspace(min(average_speeds), max(average_speeds), 20)

            heatmap, _, _ = np.histogram2d(total_distances, average_speeds, bins=[x_bins, y_bins],
                                           weights=dis_predicted_energies)
            counts, _, _ = np.histogram2d(total_distances, average_speeds, bins=[x_bins, y_bins])

            # Avoid division by zero
            average_heatmap = np.divide(heatmap, counts, where=counts != 0)

            # Mask the zero values
            average_heatmap = np.ma.masked_where(counts == 0, average_heatmap)

            etamin = 2
            etamax = 11
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(x_bins, y_bins, average_heatmap.T, shading='auto', cmap='coolwarm', vmin=etamin, vmax=etamax)
            cb = plt.colorbar(label='Average Predicted Energy Efficiency (km/kWh)')

            plt.xlabel('Trip Distance (km)')
            plt.ylabel('Average Speed (km/h)')
            plt.title(
                f"{selected_car} : Trip Distance vs. Average Speed with Energy Efficiency, {len(sample_files_dict)} files")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.show()