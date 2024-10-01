import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import random
import shap
import plotly.graph_objects as go
from scipy.stats import binned_statistic_2d
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
                plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.95))
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
            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')
            # 범례와 타이틀
            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.96))
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

def plot_contour(X, y_pred, scaler, selected_car, terminology, num_grids=400, min_count=5, save_path = None):
    if X.shape[1] != 2:
        raise ValueError("Error: X should have 2 columns.")

    # Inverse transform to original scale
    X_orig = scaler.inverse_transform(np.hstack([X, np.zeros((X.shape[0], scaler.scale_.shape[0] - 2))]))  # 원래 스케일로 역변환

    # Convert speed to km/h (assuming first feature is 'speed')
    X_orig[:, 0] *= 3.6

    # Create grid
    grid_x = np.linspace(X_orig[:, 0].min(), X_orig[:, 0].max(), num_grids)
    grid_y = np.linspace(X_orig[:, 1].min(), X_orig[:, 1].max(), num_grids)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata((X_orig[:, 0], X_orig[:, 1]), y_pred, (grid_x, grid_y), method='linear')

    # Compute density (number of points per grid cell)
    x_edges = np.linspace(X_orig[:, 0].min(), X_orig[:, 0].max(), num_grids + 1)
    y_edges = np.linspace(X_orig[:, 1].min(), X_orig[:, 1].max(), num_grids + 1)
    count_stat, _, _, _ = binned_statistic_2d(
        X_orig[:, 0], X_orig[:, 1], None, statistic='count', bins=[x_edges, y_edges]
    )

    # Mask grid_z where count_stat is below the threshold
    mask = count_stat < min_count
    grid_z[mask] = np.nan  # 또는 다른 마스킹 방법을 사용할 수 있습니다.

    # Contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title(f'{selected_car} : Contour Plot of {terminology}')

    # **고해상도로 저장**
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

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


def plot_shap_values(model, X_train, feature_names, save_path=None):
    """
    Plot SHAP values for the best XGBoost model using interventional perturbation.

    Parameters:
    model: The best trained XGBoost model.
    X_train: Training data used to calculate SHAP values.
    feature_names: List of feature names in the desired order.
    save_path: Optional. If provided, the plot will be saved to this path.
    """
    # Ensure X_train is a DataFrame with the correct feature names
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train = X_train[feature_names]

    # Use SHAP with interventional perturbation
    explainer = shap.TreeExplainer(model, feature_perturbation='interventional')

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_train, check_additivity=False)

    # Plot the SHAP summary plot with sort=False to preserve feature order
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, sort=False, show=False)

    # Get current axes
    ax = plt.gca()

    # Set black border for the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    # SHAP summary_plot은 여러 컬렉션을 생성할 수 있으므로 색상 막대를 정확히 가져와야 합니다.
    # 일반적으로 첫 번째 컬렉션이 점들의 컬렉션인 경우가 많습니다.
    try:
        colorbar = ax.collections[0].colorbar  # 첫 번째 컬렉션이 점들의 컬렉션인 경우
        if colorbar is None and len(ax.collections) > 1:
            colorbar = ax.collections[1].colorbar  # 두 번째 컬렉션을 시도
    except AttributeError:
        colorbar = None

    if colorbar:
        # 색상 막대의 위치와 범위 설정
        cbar = colorbar
        cbar.set_label('Feature Value', fontsize=12)  # 색상 막대 레이블 설정

        # 색상 막대에 수치 라벨 추가
        num_labels = 5
        ticks = np.linspace(cbar.vmin, cbar.vmax, num_labels)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])

        # 글꼴 크기 조정
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Feature Value', fontsize=12)


    # 글꼴 크기 및 레이아웃 조정
    plt.xlabel('SHAP Value', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('SHAP Summary Plot', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 레이아웃 조정
    plt.tight_layout()

    # **고해상도로 저장**
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()


def plot_composite_contour(
        X_train, y_pred_train,
        X_test, y_pred_test1, y_pred_test2,
        scaler, selected_car,
        terminology1='Train Residual',
        terminology2='Residual[1]',
        terminology3='Residual[2]',
        num_grids=30,
        min_count=10,
        save_directory=r"C:\Users\BSL\Desktop\Figures",
):
    """
    Creates a composite figure with three contour plots labeled A, B, and C.

    Parameters:
    - X_train (np.ndarray): Training features with two columns ['speed', 'acceleration'].
    - y_pred_train (np.ndarray): Predicted values for training data.
    - X_test (np.ndarray): Test features with two columns ['speed', 'acceleration'].
    - y_pred_test1 (np.ndarray): First set of predicted values for test data.
    - y_pred_test2 (np.ndarray): Second set of predicted values for test data.
    - scaler (sklearn.preprocessing scaler): Scaler used for inverse transformation.
    - selected_car (str): Identifier for the selected car.
    - terminology1 (str): Title for the first subplot.
    - terminology2 (str): Title for the second subplot.
    - terminology3 (str): Title for the third subplot.
    - num_grids (int): Number of grid points for the contour.
    - min_count (int): Minimum number of points per grid cell to display.
    - save_directory (str): Directory to save the composite figure.
    - composite_filename (str): Filename for the composite figure. If None, a default name is used.
    """

    def create_contour(ax, X, y_pred, terminology):
        if X.shape[1] != 2:
            raise ValueError("Error: X should have 2 columns.")

        # Inverse transform to original scale
        X_orig = scaler.inverse_transform(
            np.hstack([X, np.zeros((X.shape[0], scaler.scale_.shape[0] - 2))])
        )

        # Convert speed to km/h (assuming first feature is 'speed')
        X_orig[:, 0] *= 3.6

        # Create grid
        grid_x = np.linspace(X_orig[:, 0].min(), X_orig[:, 0].max(), num_grids)
        grid_y = np.linspace(X_orig[:, 1].min(), X_orig[:, 1].max(), num_grids)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        grid_z = griddata((X_orig[:, 0], X_orig[:, 1]), y_pred, (grid_x, grid_y), method='linear')

        # Compute density (number of points per grid cell)
        x_edges = np.linspace(X_orig[:, 0].min(), X_orig[:, 0].max(), num_grids + 1)
        y_edges = np.linspace(X_orig[:, 1].min(), X_orig[:, 1].max(), num_grids + 1)
        count_stat, _, _, _ = binned_statistic_2d(
            X_orig[:, 0], X_orig[:, 1], None, statistic='count', bins=[x_edges, y_edges]
        )

        # Mask grid_z where count_stat is below the threshold
        mask = count_stat < min_count
        grid_z[mask] = np.nan

        # Contour plot
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis')
        ax.set_xlabel('Speed (km/h)', fontsize=12)
        ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
        ax.set_title(f'{terminology}', fontsize=14)

        return contour

    # Ensure save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Define plot configurations
    plot_configs = [
        {
            'X': X_train,
            'y_pred': y_pred_train,
            'terminology': terminology1,
            'label': 'D'
        },
        {
            'X': X_test,
            'y_pred': y_pred_test1,
            'terminology': terminology2,
            'label': 'E'
        },
        {
            'X': X_test,
            'y_pred': y_pred_test2,
            'terminology': terminology3,
            'label': 'F'
        }
    ]

    # List to store contour objects for the colorbar
    contour_plots = []

    for ax, config in zip(axes, plot_configs):
        contour = create_contour(ax, config['X'], config['y_pred'], config['terminology'])
        contour_plots.append(contour)

        # Add label (A, B, C) to the subplot
        ax.text(
            -0.1, 1.05, config['label'],
            transform=ax.transAxes,
            fontsize=14,
            weight='bold',
            ha='left',
            va='bottom'
        )

    # Add a single colorbar for all subplots
    fig.colorbar(contour_plots[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='Prediction')

    # Save the composite figure
    save_path = os.path.join(save_directory, f"Figure8_{selected_car}_Composite.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()