import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import plotly.graph_objects as go
from GS_Functions import calculate_rrmse, calculate_rmse, calculate_mape
from scipy.interpolate import griddata
from scipy.stats import linregress
from tqdm import tqdm

def plot_power(file_lists, selected_car, target):
    for file in tqdm(file_lists):
        data = pd.read_csv(file)

        # Device Number 및 Trip Number 추출
        parts = file.split(os.sep)
        file_name = parts[-1]
        name_parts = file_name.split('_')
        trip_info = (name_parts[2] if 'altitude' in name_parts else name_parts[1]).split('.')[0]

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60
        date = t.iloc[0].strftime('%Y-%m-%d')

        power_data = np.array(data['Power_data']) / 1000
        power_phys = np.array(data['Power_phys']) / 1000
        power_diff = power_data - power_phys

        if 'Power_hybrid' in data.columns:
            power_hybrid = np.array(data['Power_hybrid']) / 1000
        
        if target == 'stacked':
            # Check if the necessary columns exist in the data
            if all(col in data.columns for col in ['A', 'B', 'C', 'D', 'E', 'F']):
                A = data['A'] / 1000
                B = data['B'] / 1000
                C = data['C'] / 1000
                D = data['D'] / 1000
                E = data['E'] / 1000
                F = data['F'] / 1000

                plt.figure(figsize=(12, 6))

                # Use lighter colors for the stackplot
                plt.stackplot(t_min, A, B, C, D, E, F,
                              labels=['A (First)', 'B (Second)', 'C (Third)', 'D (Accel)', 'E (Aux,Idle)', 'F (Altitude)'],
                              edgecolor=None,
                              colors=['#D3D3D3', '#ADD8E6', '#90EE90', '#FFB6C1', '#FFA07A', '#FFD700'])


                plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                         verticalalignment='top', horizontalalignment='right', color='black')
                plt.text(0.01, 0.99, f'{selected_car}: ' + trip_info, transform=plt.gca().transAxes, fontsize=12,
                         verticalalignment='top', horizontalalignment='left', color='black')

                plt.title('Power Stacked Graph Term by Term')
                plt.xlabel('Time (minutes)')
                plt.ylabel('Power (kW)')
                plt.legend(loc='upper left')
                plt.show()
            else:
                print(f"Error: Missing required columns in file {file_name}")

        elif target == 'physics':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Power and Physics Model Power (kW)')
            plt.plot(t_min, power_data, label='Data Power (kW)', color='tab:blue', alpha=0.6)

            # Add date and file name
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Physics Model Power')
            plt.tight_layout()
            plt.show()

        elif target == 'data':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Power (kW)')
            plt.plot(t_min, power_data, label='Data Power (kW)', color='tab:blue', alpha=0.6)

            # Add date and file name
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Data Power')
            plt.tight_layout()
            plt.show()

        elif target == 'comparison':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Power and Physics Model Power (kW)')
            plt.plot(t_min, power_data, label='Data Power (kW)', color='tab:blue', alpha=0.6)
            plt.plot(t_min, power_phys, label='Physics Model Power (kW)', color='tab:red', alpha=0.6)
            plt.ylim([-100, 100])

            # Add date and file name
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Data Power vs. Physics Model Power')
            plt.tight_layout()
            plt.show()

        elif target == 'hybrid':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Power and Physics Model Power (kW)')
            plt.plot(t_min, power_data, label='Data Power (kW)', color='tab:blue', alpha=0.6)
            plt.plot(t_min, power_phys, label='Physics Model Power (kW)', color='tab:red', alpha=0.6)
            plt.plot(t_min, power_hybrid, label='Hybrid Model Power (kW)', color='tab:green', alpha=0.6)
            plt.ylim([-100, 100])

            # Add date and file name
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Data Power vs. Physics Model Power and hybrid Model Power')
            plt.tight_layout()
            plt.show()
            
        elif target == 'altitude' and 'altitude' in data.columns:
            # 고도 차이 계산 (마지막 값은 0으로 설정)
            d_altitude = np.diff(data['altitude'])
            d_altitude = np.append(d_altitude, 0)

            # 그래프 그리기
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # 첫 번째 y축 (왼쪽): 에너지 데이터
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Power (kW)')
            ax1.plot(t_min, power_data, label='Data Power (kW)', color='tab:blue', alpha=0.6)
            ax1.plot(t_min, power_phys, label='Physics Model Power (kW)', color='tab:red', alpha=0.6)
            ax1.tick_params(axis='y')

            # 두 번째 y축 (오른쪽): 고도 데이터
            ax2 = ax1.twinx()
            ax2.set_ylabel('Altitude (m)', color='tab:green')
            ax2.set_ylim([-2, 2])
            ax2.step(t_min, d_altitude, label='Delta Altitude (m)', color='tab:green', where='mid', alpha=0.6)
            ax2.tick_params(axis='y', labelcolor='tab:green')

            # 파일과 날짜 추가
            fig.text(0.99, 0.01, date, horizontalalignment='right', color='black', fontsize=12)
            fig.text(0.01, 0.99, f'{selected_car}: ' + trip_info, verticalalignment='top', color='black', fontsize=12)

            # 범례와 타이틀
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            plt.title('Data Power vs. Physics Model Power and Delta Altitude')

            # 그래프 출력
            plt.tight_layout()
            plt.show()

        else:
            print("Invalid Target")
            return

def plot_energy(file_lists, selected_car, target):
    for file in tqdm(file_lists):
        data = pd.read_csv(file)

        # Device Number 및 Trip Number 추출
        parts = file.split(os.sep)
        file_name = parts[-1]
        name_parts = file_name.split('_')
        trip_info = (name_parts[2] if 'altitude' in name_parts else name_parts[1]).split('.')[0]

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        power_data = np.array(data['Power_data'])
        energy_data = power_data * t_diff / 3600 / 1000
        energy_data_cumulative = energy_data.cumsum()

        if 'Power_phys' in data.columns:
            power_phys = np.array(data['Power_phys'])
            energy_phys = power_phys * t_diff / 3600 / 1000
            energy_phys_cumulative = energy_phys.cumsum()

        if 'Power_hybrid' in data.columns:
            power_hybrid = np.array(data['Power_hybrid'])
            energy_hybrid = power_hybrid * t_diff / 3600 / 1000
            energy_hybrid_cumulative = energy_hybrid.cumsum()

        else:
            pass

        if target == 'physics':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Physics Model Energy (kWh)')
            plt.plot(t_min, energy_phys_cumulative, label='Physics Model Energy (kWh)', color='tab:red', alpha=0.6)

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.96))
            plt.title('Physics Model Energy')
            plt.tight_layout()
            plt.show()

        elif target == 'data':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Energy (kWh)')
            plt.plot(t_min, energy_data_cumulative, label='Data Energy (kWh)', color='tab:blue', alpha=0.6)

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

        elif target == 'hybrid':
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('BMS Energy and Physics Model Energy (kWh)')
            plt.plot(t_min, energy_phys_cumulative, label='Physics Model Energy (kWh)', color='tab:red', alpha=0.6)
            plt.plot(t_min, energy_data_cumulative, label='Data Energy (kWh)', color='tab:blue', alpha=0.6)
            plt.plot(t_min, energy_hybrid_cumulative, label='Hybrid Model Energy (kWh)', color='tab:green', alpha=0.6)

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.96))
            plt.title('Physics Model Energy vs. Data Energy and Hybrid Model Energy')
            plt.tight_layout()
            plt.show()

        elif target == 'comparison':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('BMS Energy and Physics Model Energy (kWh)')
            plt.plot(t_min, energy_phys_cumulative, label='Physics Model Energy (kWh)', color='tab:red', alpha=0.6)
            plt.plot(t_min, energy_data_cumulative, label='Data Energy (kWh)', color='tab:blue', alpha=0.6)
            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.96))
            plt.title('Physics Model Energy vs. BMS Energy')
            plt.tight_layout()
            plt.show()

        elif target == 'altitude' and 'altitude' in data.columns:
            # 고도 데이터
            altitude = np.array(data['altitude'])

            # 그래프 그리기
            fig, ax1 = plt.subplots(figsize=(10, 6))
            # 첫 번째 y축 (왼쪽): 에너지 데이터
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Energy (kWh)')
            ax1.plot(t_min, energy_phys_cumulative, label='Physics Model Energy (kWh)', color='tab:red', alpha=0.6)
            ax1.plot(t_min, energy_data_cumulative, label='Data Energy (kWh)', color='tab:blue', alpha=0.6)
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
            fig.text(0.01, 0.99, f'{selected_car}: ' + trip_info, verticalalignment='top', color='black', fontsize=12)
            # 범례와 타이틀
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            plt.title('Physics Model Energy vs. Data Energy and Altitude')
            # 그래프 출력
            plt.tight_layout()
            plt.show()

        else:
            print("Invalid Target")
            return
def plot_energy_scatter(file_lists, selected_car, target):
    energies_data = []
    energies_phys = []
    energies_hybrid = []

    all_energies_data = []
    all_energies_phys = []
    all_energies_hybrid = []

    # calculate total energy using whole data
    for file in tqdm(file_lists):
        data = pd.read_csv(file)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        power_data = np.array(data['Power_data'])
        energy_data = power_data * t_diff / 3600 / 1000
        all_energies_data.append(energy_data.cumsum()[-1])

        if 'Power_phys' in data.columns:
            power_phys = np.array(data['Power_phys'])
            energy_phys = power_phys * t_diff / 3600 / 1000
            all_energies_phys.append(energy_phys.cumsum()[-1])

        if 'Power_hybrid' in data.columns:
            power_hybrid = np.array(data['Power_hybrid'])
            energy_hybrid = power_hybrid * t_diff / 3600 / 1000
            all_energies_hybrid.append(energy_hybrid.cumsum()[-1])

    # select 1000 samples in random
    sample_size = min(1000, len(file_lists))
    sampled_files = random.sample(file_lists, sample_size)

    for file in tqdm(sampled_files):
        data = pd.read_csv(file)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        power_data = np.array(data['Power_data'])
        energy_data = power_data * t_diff / 3600 / 1000
        energies_data.append(energy_data.cumsum()[-1])

        if 'Power_phys' in data.columns:
            power_phys = np.array(data['Power_phys'])
            energy_phys = power_phys * t_diff / 3600 / 1000
            energies_phys.append(energy_phys.cumsum()[-1])

        if 'Power_hybrid' in data.columns:
            power_hybrid = np.array(data['Power_hybrid'])
            energy_hybrid = power_hybrid * t_diff / 3600 / 1000
            energies_hybrid.append(energy_hybrid.cumsum()[-1])

    if target == 'physics':
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = cm.rainbow(np.linspace(0, 1, len(energies_data)))

        ax.set_xlabel('Data Energy (kWh)')
        ax.set_ylabel('Physics_based Model Energy (kWh)')

        for i in range(len(energies_phys)):
            ax.scatter(energies_data[i], energies_phys[i], color=colors[i], facecolors='none',
                       edgecolors=colors[i], label='Model Energy' if i == 0 else "")

        # 전체 데이터셋을 사용하여 45도 기준선 계산
        slope_original, intercept_original, _, _, _ = linregress(all_energies_data, all_energies_phys)
        ax.plot(np.array(energies_data), intercept_original + slope_original * np.array(energies_data),
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
        rmse, relative_rmse = calculate_rmse(all_energies_data, all_energies_phys), calculate_rrmse(all_energies_data, all_energies_phys)
        plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}kWh\nRelative RMSE: {relative_rmse:.2%}',
                 transform=ax.transAxes, fontsize=12, verticalalignment='top')

        plt.legend()
        plt.title(f"{selected_car} : BMS Energy vs. Model Energy")
        plt.show()

    elif target == 'hybrid':
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = cm.rainbow(np.linspace(0, 1, len(energies_data)))

        ax.set_xlabel('Data Energy (kWh)')
        ax.set_ylabel('Hybrid Model Energy (kWh)')

        for i in range(len(energies_data)):
            ax.scatter(energies_data[i], energies_phys[i], color=colors[i], facecolors='none',
                       edgecolors=colors[i], label='Before learning' if i == 0 else "")

        for i in range(len(energies_data)):
            ax.scatter(energies_data[i], energies_hybrid[i], color=colors[i],
                       label='After learning' if i == 0 else "")

        slope_original, intercept_original, _, _, _ = linregress(all_energies_data, all_energies_phys)
        ax.plot(np.array(energies_data), intercept_original + slope_original * np.array(energies_data),
                color='lightblue')

        slope, intercept, _, _, _ = linregress(all_energies_data, all_energies_hybrid)
        ax.plot(np.array(energies_data), intercept + slope * np.array(energies_data), 'b')

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        # RMSE & NRMSE
        mape_before, relative_rmse_before = calculate_mape(np.array(all_energies_data), np.array(all_energies_phys)), calculate_rrmse(np.array(all_energies_data), np.array(all_energies_phys))
        mape_after, relative_rmse_after = calculate_mape(np.array(all_energies_data), np.array(all_energies_hybrid)), calculate_rrmse(np.array(all_energies_data), np.array(all_energies_hybrid))
        plt.text(0.6, 0.15, f'MAPE (Before): {mape_before:.2f}%\nRRMSE (Before): {relative_rmse_before:.2%}\nMAPE (After): {mape_after:.2f}%\nRRMSE (After): {relative_rmse_after:.2%}',
                 transform=ax.transAxes, fontsize=10, verticalalignment='top')

        plt.legend()
        plt.title(f"{selected_car} : BMS Energy vs. Trained Model Energy")
        plt.show()

    else:
        print('Invalid Target')
        return

def plot_driver_energy_scatter(vehicle_files, selected_car):
    energies_data = {}
    energies_phys = {}
    energies_hybrid = {}

    colors = cm.rainbow(np.linspace(0, 1, len(vehicle_files)))
    color_map = {}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel('Data Energy (kWh)')
    ax.set_ylabel('Hybrid Model Energy (kWh)')

    for i, (id, files) in enumerate(vehicle_files.items()):
        energies_data[id] = []
        energies_phys[id] = []
        energies_hybrid[id] = []
        color_map[id] = colors[i]

        for file in tqdm(files, desc=f'Processing {id}'):
            data = pd.read_csv(file)

            t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            t_diff = t.diff().dt.total_seconds().fillna(0)
            t_diff = np.array(t_diff.fillna(0))

            power_data = np.array(data['Power_data'])
            energy_data = power_data * t_diff / 3600 / 1000
            energies_data[id].append(energy_data.cumsum()[-1])

            if 'Power_phys' in data.columns:
                power_phys = np.array(data['Power_phys'])
                energy_phys = power_phys * t_diff / 3600 / 1000
                energies_phys[id].append(energy_phys.cumsum()[-1])

            if 'Power_hybrid' in data.columns:
                power_hybrid = np.array(data['Power_hybrid'])
                predicted_energy = power_hybrid * t_diff / 3600 / 1000
                energies_hybrid[id].append(predicted_energy.cumsum()[-1])

        ax.scatter(energies_data[id], energies_hybrid[id], facecolors='none', edgecolors=color_map[id],
                   label=f'{id} After learning')

        slope, intercept, _, _, _ = linregress(energies_data[id], energies_hybrid[id])
        ax.plot(np.array(energies_data[id]), intercept + slope * np.array(energies_data[id]), color=color_map[id])

        # Calculate RMSE & NRMSE for each id
        mape_before, relative_rmse_before = calculate_mape(np.array(energies_data[id]), np.array(energies_phys[id])), calculate_rrmse(np.array(energies_data[id]), np.array(energies_phys[id]))
        mape_after, relative_rmse_after = calculate_mape(np.array(energies_data[id]), np.array(energies_hybrid[id])), calculate_rrmse(np.array(energies_data[id]), np.array(energies_hybrid[id]))
        ax.text(0.05, 0.95 - i * 0.1, f'{id}\nMAPE (Before): {mape_before:.2f}%\nRRMSE (Before): {relative_rmse_before:.2%}\nMAPE (After): {mape_after:.2f}%\nRRMSE (After): {relative_rmse_after:.2%}',
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
        power_data = np.array(data['Power_data'])
        power_phys = np.array(data['Power_phys'])
        diff_power = power_data - power_phys

        # Plotting for each file
        plt.figure(figsize=(10, 6))
        plt.scatter(d_altitude, diff_power, alpha=0.6)  # alpha for transparency

        plt.xlabel('Delta Altitude')
        plt.xlim(-2, 2)
        plt.ylabel('Power Difference (Data - Model)')
        plt.ylim(-40000, 40000)
        plt.title(f'Delta Altitude vs Power Difference for {file}')
        plt.grid(True)
        plt.show()

def plot_energy_dis(file_lists, selected_car, Target):
    dis_energies_phys = []
    dis_energies_data = []
    dis_energies_hybrid = []
    total_distances = []

    # Official fuel efficiency data (km/kWh)
    official_efficiency = {
        'Ioniq5': [4.667, 5.371],
        'EV6': [4.524, 5.515],
        'NiroEV': [5.277],
        'KonaEV': [5.655],
        'Ioniq6': [4.855, 6.597],
        'GV60': [4.241, 5.277]
    }

    for file in tqdm(file_lists):
        data = pd.read_csv(file)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        v = data['speed']
        v = np.array(v)

        distance = v * t_diff
        total_distance = distance.cumsum()

        if 'Power_phys' in data.columns:
            power_phys = np.array(data['Power_phys'])
            energy_phys = power_phys * t_diff / 3600 / 1000
        else:
            energy_phys = np.zeros_like(t_diff)

        Power_data = np.array(data['Power_data'])
        energy_data = Power_data * t_diff / 3600 / 1000

        if 'Power_hybrid' in data.columns:
            power_hybrid = np.array(data['Power_hybrid'])
            energy_hybrid = power_hybrid * t_diff / 3600 / 1000
            dis_energy_hybrid = ((total_distance[-1] / 1000) / (energy_hybrid.cumsum()[-1])) if energy_hybrid.cumsum()[-1] != 0 else 0
            dis_energies_hybrid.append(dis_energy_hybrid)

        dis_energy_phys = ((total_distance[-1] / 1000) / (energy_phys.cumsum()[-1])) if energy_phys.cumsum()[-1] != 0 else 0
        dis_energies_phys.append(dis_energy_phys)

        dis_data_energy = ((total_distance[-1] / 1000) / (energy_data.cumsum()[-1])) if energy_data.cumsum()[-1] != 0 else 0
        dis_energies_data.append(dis_data_energy)

        total_distances.append(total_distance[-1])

    # Function to add the official efficiency lines and shaded area
    def add_efficiency_lines(selected_car):
        if selected_car in official_efficiency:
            eff_range = official_efficiency[selected_car]
            if len(eff_range) == 2:
                # Shade between the efficiency range
                plt.fill_betweenx(plt.gca().get_ylim(), eff_range[0], eff_range[1], color='green', alpha=0.3, hatch='/')
                plt.text(eff_range[1] + 0.05, plt.gca().get_ylim()[1] * 0.8, f'Official Mileage', color='green', fontsize=12, alpha=0.7)
            else:
                # For single value efficiency, draw a line
                plt.axvline(eff_range[0], color='green', linestyle='--', alpha=0.5)
                plt.text(eff_range[0] + 0.05, plt.gca().get_ylim()[1] * 0.8, f'Official Mileage', color='green', alpha=0.7, fontsize=10)

    if Target == 'physics':
        mean_value = np.mean(dis_energies_phys)
        hist_data = sns.histplot(dis_energies_phys, bins='auto', color='gray', kde=False)
        plt.axvline(mean_value, color='red', linestyle='--')
        plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red', fontsize=12)
        total_samples = len(dis_energies_phys)
        plt.text(0.95, 0.95, f'Total Samples: {total_samples}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, color='black')
        plt.xlim(0, 25)
        plt.xlabel('Total Distance / Total Model Energy (km/kWh)')
        plt.ylabel('Number of trips')
        plt.title(f"{selected_car} : Total Distance / Total Model Energy Distribution")
        add_efficiency_lines(selected_car)
        plt.grid(False)
        plt.show()

    elif Target == 'data':
        mean_value = np.mean(dis_energies_data)
        hist_data = sns.histplot(dis_energies_data, bins='auto', color='gray', kde=False)
        plt.axvline(mean_value, color='red', linestyle='--')
        plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red', fontsize=12)
        total_samples = len(dis_energies_data)
        plt.text(0.95, 0.95, f'Total Samples: {total_samples}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, color='black')
        plt.xlim(0, 25)
        plt.xlabel('Total Distance / Total Data Energy (km/kWh)')
        plt.ylabel('Number of trips')
        plt.title(f"{selected_car} : Total Distance / Total Data Energy Distribution")
        add_efficiency_lines(selected_car)
        plt.grid(False)
        plt.show()

    elif Target == 'hybrid' and 'Power_hybrid' in data.columns:
        mean_value = np.mean(dis_energies_hybrid)
        hist_data = sns.histplot(dis_energies_hybrid, bins='auto', color='gray', kde=False)
        plt.axvline(mean_value, color='red', linestyle='--')
        plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red', fontsize=12)
        total_samples = len(dis_energies_hybrid)
        plt.text(0.95, 0.95, f'Total Samples: {total_samples}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, color='black')
        plt.xlim(0, 25)
        plt.xlabel('Total Distance / Total Hybrid Model Energy (km/kWh)')
        plt.ylabel('Number of trips')
        plt.title(f"{selected_car} : Total Distance / Total Hybrid Model Energy Distribution")
        add_efficiency_lines(selected_car)
        plt.grid(False)
        plt.show()

    else:
        print("Invalid Target. Please try again.")
        return

def plot_3d(X, y_true, y_pred, fold_num, vehicle, scaler, num_grids=400, samples_per_grid=30, output_file=None):
    if X.shape[1] != 2:
        print("Error: X should have 2 columns.")
        return

    # Inverse transform to original scale
    X_orig = scaler.inverse_transform(X)

    # Convert Speed Unit (km/h)
    X_orig[:, 0] *= 3.6

    # calculate grid size
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

            # select index in grid
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

def plot_contour(X, y_pred, scaler, selected_car, terminology, num_grids=400):
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
    plt.title(f'{selected_car} : Contour Plot of {terminology}')
    plt.show()


#
# def plot_contour(X, y_pred, scaler, selected_car, terminology, num_grids=400, min_samples=5):
#     if X.shape[1] != 2:
#         raise ValueError("Error: X should have 2 columns.")
#
#     # Inverse transform to original scale
#     X_orig = scaler.inverse_transform(X)
#
#     # Convert speed to km/h
#     X_orig[:, 0] *= 3.6
#
#     # Create grid
#     grid_x = np.linspace(X_orig[:, 0].min(), X_orig[:, 0].max(), num_grids)
#     grid_y = np.linspace(X_orig[:, 1].min(), X_orig[:, 1].max(), num_grids)
#     grid_x, grid_y = np.meshgrid(grid_x, grid_y)
#
#     # Create empty grid to hold the counts
#     sample_count = np.zeros_like(grid_x)
#
#     # Calculate grid cell size
#     grid_x_size = np.mean(np.diff(grid_x[0]))
#     grid_y_size = np.mean(np.diff(grid_y[:, 0]))
#
#     # Count the number of samples in each grid cell
#     for i in range(len(X_orig)):
#         x_idx = np.searchsorted(grid_x[0], X_orig[i, 0]) - 1
#         y_idx = np.searchsorted(grid_y[:, 0], X_orig[i, 1]) - 1
#
#         if 0 <= x_idx < num_grids and 0 <= y_idx < num_grids:
#             sample_count[y_idx, x_idx] += 1
#
#     # Initialize grid_z with NaNs
#     grid_z = np.full_like(grid_x, np.nan)
#
#     # Assign values to grid_z only if sample count meets the minimum requirement
#     for i in range(num_grids):
#         for j in range(num_grids):
#             if sample_count[i, j] >= min_samples:
#                 grid_z[i, j] = griddata((X_orig[:, 0], X_orig[:, 1]), y_pred, (grid_x[i, j], grid_y[i, j]),
#                                         method='linear')
#
#     # Contour plot
#     plt.figure(figsize=(10, 8))
#     contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis')
#     plt.colorbar(contour)
#     plt.xlabel('Speed (km/h)')
#     plt.ylabel('Acceleration (m/s²)')
#     plt.title(f'{selected_car} : Contour Plot of {terminology}')
#     plt.show()
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
    Residual = merged_data['Power_data'] - merged_data['Power_phys']

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
            dis_energies_data = []
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

                power_data = np.array(data['Power_data'])
                energy_data = power_data * t_diff / 3600 / 1000

                total_energy = energy_data.cumsum()[-1]
                dis_energy_data = ((total_distance / 1000) / total_energy) if total_energy != 0 else 0

                total_distances.append(total_distance / 1000)  # convert to km
                dis_energies_data.append(dis_energy_data)
                average_speed = np.mean(v) * 3.6  # converting m/s to km/h
                average_speeds.append(average_speed)

            # Create bins
            x_bins = np.linspace(min(total_distances), max(total_distances), 20)
            y_bins = np.linspace(min(average_speeds), max(average_speeds), 20)

            heatmap, _, _ = np.histogram2d(total_distances, average_speeds, bins=[x_bins, y_bins], weights=dis_energies_data)
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
        dis_energies_data = []
        dis_energies_phys = []
        dis_energies_hybrid = []
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

            if 'Power_phys' in data.columns:
                power_phys = np.array(data['Power_phys'])
                energy_model = power_phys * t_diff / 3600 / 1000
            else:
                energy_model = np.zeros_like(t_diff)

            power_data = np.array(data['Power_data'])
            energy_data = power_data * t_diff / 3600 / 1000

            if 'Power_hybrid' in data.columns:
                power_hybrid = data['Power_hybrid']
                power_hybrid = np.array(power_hybrid)
                energy_hybrid = power_hybrid * t_diff / 3600 / 1000
                dis_energy_hybrid = ((total_distance / 1000) / (energy_hybrid.cumsum()[-1])) if \
                    energy_hybrid.cumsum()[-1] != 0 else 0
                dis_energies_hybrid.append(dis_energy_hybrid)

            # calculate Total distance / Total Energy for each file (if Total Energy is 0, set the value to 0)
            dis_energy_phys = ((total_distance / 1000) / (energy_model.cumsum()[-1])) if energy_model.cumsum()[
                                                                                                -1] != 0 else 0
            dis_energies_phys.append(dis_energy_phys)

            dis_energy_data = ((total_distance / 1000) / (energy_data.cumsum()[-1])) if energy_data.cumsum()[
                                                                                                -1] != 0 else 0
            dis_energies_data.append(dis_energy_data)

            total_distances.append(total_distance / 1000)

            average_speed = np.mean(v) * 3.6  # converting m/s to km/h
            average_speeds.append(average_speed)
        if Target == 'physics' :
            # Create bins
            x_bins = np.linspace(min(total_distances), max(total_distances), 20)
            y_bins = np.linspace(min(average_speeds), max(average_speeds), 20)

            heatmap, _, _ = np.histogram2d(total_distances, average_speeds, bins=[x_bins, y_bins], weights=dis_energies_phys)
            counts, _, _ = np.histogram2d(total_distances, average_speeds, bins=[x_bins, y_bins])

            # Avoid division by zero
            average_heatmap = np.divide(heatmap, counts, where=counts != 0)

            # Mask the zero values
            average_heatmap = np.ma.masked_where(counts == 0, average_heatmap)

            etamin = 2
            etamax = 11
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(x_bins, y_bins, average_heatmap.T, shading='auto', cmap='coolwarm', vmin=etamin, vmax=etamax)
            cb = plt.colorbar(label='Average Physics model Energy Efficiency (km/kWh)')

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
                                           weights=dis_energies_data)
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
        elif Target == 'hybrid':
            # Create bins
            x_bins = np.linspace(min(total_distances), max(total_distances), 20)
            y_bins = np.linspace(min(average_speeds), max(average_speeds), 20)

            heatmap, _, _ = np.histogram2d(total_distances, average_speeds, bins=[x_bins, y_bins],
                                           weights=dis_energies_hybrid)
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