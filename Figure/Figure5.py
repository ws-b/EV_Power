import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import plotly.graph_objects as go
from GS_Functions import get_vehicle_files, calculate_rmse, calculate_mape, calculate_rrmse
from scipy.interpolate import griddata
from scipy.stats import linregress
from tqdm import tqdm
from GS_vehicle_dict import vehicle_dict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_file_lists(directory):
    vehicle_files = {vehicle: [] for vehicle in vehicle_dict.keys()}

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Match filename with vehicle IDs
            for vehicle, ids in vehicle_dict.items():
                if any(vid in filename for vid in ids):
                    vehicle_files[vehicle].append(os.path.join(directory, filename))
                    break  # Stop searching once a match is found

    return vehicle_files


# Example usage of the function
directory = r"D:\SamsungSTF\Processed_Data\TripByTrip"
vehicle_files = get_file_lists(directory)
selected_cars = ['EV6', 'Ioniq5']

#save_path
fig_save_path = r"C:\Users\BSL\Desktop\Figures"

def figure5(vehicle_files, selected_cars):
    # Initialize dictionaries for storing data for selected vehicles
    energies_dict = {car: {'data': [], 'phys': [], 'hybrid': []} for car in selected_cars}
    all_energies_dict = {car: {'data': [], 'phys': [], 'hybrid': []} for car in selected_cars}

    # Calculate total energy using whole data
    for selected_car in selected_cars:

        for file in tqdm(vehicle_files[selected_car]):
            data = pd.read_csv(file)

            t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            t_diff = t.diff().dt.total_seconds().fillna(0)
            t_diff = np.array(t_diff.fillna(0))

            power_data = np.array(data['Power_data'])
            energy_data = power_data * t_diff / 3600 / 1000
            all_energies_dict[selected_car]['data'].append(energy_data.cumsum()[-1])

            if 'Power_phys' in data.columns:
                power_phys = np.array(data['Power_phys'])
                energy_phys = power_phys * t_diff / 3600 / 1000
                all_energies_dict[selected_car]['phys'].append(energy_phys.cumsum()[-1])

            if 'Power_hybrid' in data.columns:
                power_hybrid = np.array(data['Power_hybrid'])
                energy_hybrid = power_hybrid * t_diff / 3600 / 1000
                all_energies_dict[selected_car]['hybrid'].append(energy_hybrid.cumsum()[-1])

    # Select 1000 samples in random
    sample_size = min(1000, len(vehicle_files[selected_car]))
    sampled_files = {car: random.sample(vehicle_files[car], sample_size) for car in selected_cars}

    for selected_car in selected_cars:
        for file in tqdm(sampled_files[selected_car]):
            data = pd.read_csv(file)

            t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            t_diff = t.diff().dt.total_seconds().fillna(0)
            t_diff = np.array(t_diff.fillna(0))

            power_data = np.array(data['Power_data'])
            energy_data = power_data * t_diff / 3600 / 1000
            energies_dict[selected_car]['data'].append(energy_data.cumsum()[-1])

            if 'Power_phys' in data.columns:
                power_phys = np.array(data['Power_phys'])
                energy_phys = power_phys * t_diff / 3600 / 1000
                energies_dict[selected_car]['phys'].append(energy_phys.cumsum()[-1])

            if 'Power_hybrid' in data.columns:
                power_hybrid = np.array(data['Power_hybrid'])
                energy_hybrid = power_hybrid * t_diff / 3600 / 1000
                energies_dict[selected_car]['hybrid'].append(energy_hybrid.cumsum()[-1])

    # Create 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    for i, selected_car in enumerate(selected_cars):
        ax = axs[0, i]
        colors = cm.rainbow(np.linspace(0, 1, len(energies_dict[selected_car]['data'])))

        ax.set_xlabel('Data Energy (kWh)')
        ax.set_ylabel('Hybrid Model Energy (kWh)')
        ax.text(-0.1, 1.05, chr(65 + i), transform=ax.transAxes, size=14, weight='bold')

        for j in range(len(energies_dict[selected_car]['data'])):
            ax.scatter(energies_dict[selected_car]['data'][j], energies_dict[selected_car]['phys'][j], color=colors[j],
                       facecolors='none',
                       edgecolors=colors[j], label='Physics-based Model' if j == 0 else "")

        for j in range(len(energies_dict[selected_car]['data'])):
            ax.scatter(energies_dict[selected_car]['data'][j], energies_dict[selected_car]['hybrid'][j],
                       color=colors[j],
                       label='Hybrid Model' if j == 0 else "")

        # 로그 스케일에 맞춰 데이터 변환
        log_data_energy = np.log(energies_dict[selected_car]['data'])
        log_phys_energy = np.log(energies_dict[selected_car]['phys'])
        log_hybrid_energy = np.log(energies_dict[selected_car]['hybrid'])

        # Physics 모델과의 로그 스케일 회귀 분석
        slope_original, intercept_original, _, _, _ = linregress(log_data_energy, log_phys_energy)

        # Hybrid 모델과의 로그 스케일 회귀 분석
        slope, intercept, _, _, _ = linregress(log_data_energy, log_hybrid_energy)

        # 로그 스케일에 맞는 회귀선을 그리기 위해 exp() 함수로 변환
        ax.plot(np.array(energies_dict[selected_car]['data']),
                np.exp(intercept_original + slope_original * np.log(energies_dict[selected_car]['data'])),
                color='lightblue')

        ax.plot(np.array(energies_dict[selected_car]['data']),
                np.exp(intercept + slope * np.log(energies_dict[selected_car]['data'])),
                'b')
        # MAPE & RRMSE calculations for display
        mape_before = calculate_mape(np.array(energies_dict[selected_car]['data']),
                                     np.array(energies_dict[selected_car]['phys']))
        mape_after = calculate_mape(np.array(energies_dict[selected_car]['data']),
                                    np.array(energies_dict[selected_car]['hybrid']))

        # Displaying the MAPE and RRMSE values in the plot
        ax.text(0.6, 0.15,
                f'MAPE (Before): {mape_before:.2f}%\n'
                f'MAPE (After): {mape_after:.2f}%\n',
                transform=ax.transAxes, fontsize=10, verticalalignment='top')

        # 축의 범위를 설정하고 대각선 비교선을 그리기 위한 lims 설정
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

        # x축과 y축을 로그 스케일로 설정
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_aspect('equal')
        ax.set_xlim(1, None)
        ax.set_ylim(1, None)

        # Add legend for A and B
        ax.legend(loc='upper left')

        # Add subplot title
        ax.set_title(f"{selected_car} : Data Energy vs. Hybrid Model Energy")

    for i, selected_car in enumerate(selected_cars):
        # Select random sample ids for this car
        sample_ids = vehicle_dict[selected_car][0:3]
        sample_files_dict = {id: [f for f in vehicle_files[selected_car] if id in f] for id in sample_ids}

        energies_data = {}
        energies_phys = {}
        energies_hybrid = {}

        # 색상 범위를 조정하여
        colors = cm.rainbow(np.linspace(0.2, 0.8, len(sample_files_dict)))

        color_map = {}
        ax = axs[1, i]  # C and D are in the second row

        for j, (id, files) in enumerate(sample_files_dict.items()):
            energies_data[id] = []
            energies_phys[id] = []
            energies_hybrid[id] = []
            color_map[id] = colors[j]
            driver_label = f"Driver {j + 1}"
            for file in tqdm(files, desc=f'Processing {selected_car} - Driver {id}'):
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

            # 로그 스케일을 위해 0 또는 음수 값을 제외하는 필터링
            filtered_data_energy = np.array(energies_data[id])
            filtered_hybrid_energy = np.array(energies_hybrid[id])

            positive_mask = (filtered_data_energy > 0) & (filtered_hybrid_energy > 0)

            filtered_data_energy = filtered_data_energy[positive_mask]
            filtered_hybrid_energy = filtered_hybrid_energy[positive_mask]

            # Scatter plot and regression line
            ax.scatter(filtered_data_energy, filtered_hybrid_energy, facecolors='none',
                       edgecolors=color_map[id], label=f'{driver_label} Hybrid Model')

            # 로그 스케일 회귀 분석
            log_data_energy = np.log(filtered_data_energy)
            log_hybrid_energy = np.log(filtered_hybrid_energy)

            slope, intercept, _, _, _ = linregress(log_data_energy, log_hybrid_energy)
            ax.plot(filtered_data_energy, np.exp(intercept + slope * np.log(filtered_data_energy)),
                    color=color_map[id])

            # Calculate RMSE & NRMSE for each car
            mape_before = calculate_mape(np.array(energies_data[id]), np.array(energies_phys[id]))
            mape_after = calculate_mape(np.array(energies_data[id]), np.array(energies_hybrid[id]))

            ax.text(0.05, 0.95 - j * 0.07,
                    f'{selected_car} {driver_label}\nMAPE (Before): {mape_before:.2f}%\nMAPE (After): {mape_after:.2f}%',
                    transform=ax.transAxes, fontsize=8, verticalalignment='top', color=color_map[id])

        # Set subplot limits and aspects
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

        # x축과 y축을 로그 스케일로 설정
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_aspect('equal')
        ax.set_xlim(1, None)
        ax.set_ylim(1, None)

        # Add legend for C and D
        ax.legend(loc='upper right')
        # Set titles, labels, and markers for C and D
        ax.set_xlabel('Data Energy (kWh)')
        ax.set_ylabel('Hybrid Model Energy (kWh)')
        ax.text(-0.1, 1.05, chr(67 + i), transform=ax.transAxes, size=16, weight='bold')
        ax.set_title(f"{selected_car}'s Driver : Data Energy vs. Hybrid Model Energy")

    save_path = os.path.join(fig_save_path, 'figure5.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

figure5(vehicle_files, selected_cars)