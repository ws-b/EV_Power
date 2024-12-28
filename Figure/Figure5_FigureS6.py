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
from scipy.stats import linregress, t
from tqdm import tqdm
from GS_vehicle_dict import vehicle_dict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

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
selected_cars = ['KonaEV', 'GV60']

#save_path
fig_save_path = r"C:\Users\BSL\Desktop\Figures\Supplementary"

def calculate_prediction_interval(x_val, x, y, slope, intercept, confidence=0.95):
    """
    x_new: 예측 구간을 계산할 새로운 x 값 (로그 스케일)
    x: 원래의 x 데이터 (로그 스케일)
    y: 원래의 y 데이터 (로그 스케일)
    slope: 회귀 기울기
    intercept: 회귀 절편
    confidence: 신뢰 수준 (default 0.95)

    반환값:
    - y_pred_new: 예측된 y 값 (로그 스케일)
    - y_lower_new: 예측 구간 하한 (로그 스케일)
    - y_upper_new: 예측 구간 상한 (로그 스케일)
    """
    n = len(x)
    mean_x = np.mean(x)
    Sxx = np.sum((x - mean_x) ** 2)

    # 잔차 계산
    y_pred = intercept + slope * x
    residuals = y - y_pred
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))

    # t-분포의 임계값 계산
    t_val = t.ppf((1 + confidence) / 2., n - 2)

    # 예측 구간 계산
    y_pred = intercept + slope * x_val
    se_new = s_err * np.sqrt(1 + (x_val - mean_x) ** 2 / Sxx)
    y_lower = y_pred - t_val * se_new
    y_upper = y_pred + t_val * se_new

    return y_pred, y_lower, y_upper

def figure5(vehicle_files, selected_cars):
    # Initialize dictionaries for storing data for selected vehicles
    all_energies_dict = {car: {'data': [], 'phys': [], 'hybrid': []} for car in selected_cars}

    # Variables to track global min and max
    global_x_max = float('-inf')
    global_y_max = float('-inf')

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

    # After collecting all data, determine global min and max
    for car in selected_cars:
        data = np.array(all_energies_dict[car]['data'])
        phys = np.array(all_energies_dict[car]['phys'])
        hybrid = np.array(all_energies_dict[car]['hybrid'])

        global_x_max = max(global_x_max, data.max())
        global_y_max = max(global_y_max, phys.max(), hybrid.max(), global_x_max)

    # Create 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    for i, selected_car in enumerate(selected_cars):
        ax = axs[0, i]

        ax.set_xlabel('Data Energy (kWh)')
        ax.set_ylabel('Model Energy (kWh)')
        ax.text(-0.1, 1.05, chr(65 + i), transform=ax.transAxes, size=16, weight='bold')

        # log scale data
        log_data_energy = np.log(all_energies_dict[selected_car]['data'])
        log_phys_energy = np.log(all_energies_dict[selected_car]['phys'])
        log_hybrid_energy = np.log(all_energies_dict[selected_car]['hybrid'])

        # Physics 모델과의 로그 스케일 회귀 분석
        slope_original, intercept_original, _, _, _ = linregress(log_data_energy, log_phys_energy)

        # Hybrid 모델과의 로그 스케일 회귀 분석
        slope, intercept, _, _, _ = linregress(log_data_energy, log_hybrid_energy)

        # x축의 최솟값과 최댓값 정의 (원래 스케일)
        x_min = np.min(all_energies_dict[selected_car]['data'])
        x_max = np.max(all_energies_dict[selected_car]['data'])
        x_vals = np.linspace(x_min, x_max, 100)

        # 회귀선 계산 (원래 스케일에서 y = intercept * x^slope 형태)
        y_phys = np.exp(intercept_original) * x_vals ** slope_original
        y_hybrid = np.exp(intercept) * x_vals ** slope

        # 모델 예측 선 플롯 (레이블 포함)
        line_phys, = ax.plot(x_vals, y_phys, color="#efc000ff", label='Physics-based Model')
        line_hybrid, = ax.plot(x_vals, y_hybrid, color="#cd534cff", label='Hybrid Model(XGB)')


        # 예측 구간 계산 for Physics Model
        _, y_phys_lower, y_phys_upper = calculate_prediction_interval(
            np.log(x_vals), log_data_energy, log_phys_energy, slope_original, intercept_original
        )
        y_phys_lower_orig = np.exp(y_phys_lower)
        y_phys_upper_orig = np.exp(y_phys_upper)

        # 예측 구간 음영 영역 플롯 for Physics Model
        ax.fill_between(x_vals, y_phys_lower_orig, y_phys_upper_orig, color="#efc000ff", alpha=0.2, label='_nolegend_')

        # 예측 구간 계산
        _, y_hybrid_lower, y_hybrid_upper = calculate_prediction_interval(np.log(x_vals), log_data_energy, log_hybrid_energy,
                                                                          slope, intercept)
        y_hybrid_lower_orig = np.exp(y_hybrid_lower)
        y_hybrid_upper_orig = np.exp(y_hybrid_upper)

        # 예측 구간 음영 영역 플롯
        ax.fill_between(x_vals, y_hybrid_lower_orig, y_hybrid_upper_orig, color="#cd534cff", alpha=0.2, label='_nolegend_')

        # 회색 패치를 생성하여 레전드에 추가
        prediction_patch = mpatches.Patch(color="#747678ff", alpha=0.2, label='Prediction Interval 90%')

        # MAPE calculations using the entire dataset
        mape_before = calculate_mape(np.array(all_energies_dict[selected_car]['data']),
                                     np.array(all_energies_dict[selected_car]['phys']))
        mape_after = calculate_mape(np.array(all_energies_dict[selected_car]['data']),
                                    np.array(all_energies_dict[selected_car]['hybrid']))

        # Displaying the MAPE in the plot
        ax.text(0.6, 0.10,
                f'MAPE (Before): {mape_before:.2f}%\n'
                f'MAPE (After): {mape_after:.2f}%\n',
                transform=ax.transAxes, fontsize=10, verticalalignment='top')

        # 축의 범위를 설정하고 대각선 비교선을 그리기 위한 lims 설정
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        # ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

        # x축과 y축을 로그 스케일로 설정
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_aspect('equal')
        # ax.set_xlim(1, None)
        # ax.set_ylim(1, None)

        # Set x and y limits using global min and max
        ax.set_xlim(1, global_y_max * 1.05)
        ax.set_ylim(1, global_y_max * 1.05)

        # 레전드 항목을 수동으로 설정
        handles = [line_phys, line_hybrid, prediction_patch]
        labels = [handle.get_label() for handle in handles]
        ax.legend(labels=labels,handles=handles, loc='upper left')

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
        colors = [
            "#0073c2ff", "#efc000ff", "#cd534cff", "#20854eff", "#925e9fff",
            "#e18727ff", "#4dbbd5ff", "#ee4c97ff", "#7e6148ff", "#747678ff"
        ]
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
                       edgecolors=color_map[id], label=f'{driver_label} Hybrid Model(XGB)')

            # 로그 스케일 회귀 분석
            log_data_energy = np.log(filtered_data_energy)
            log_hybrid_energy = np.log(filtered_hybrid_energy)

            slope, intercept, _, _, _ = linregress(log_data_energy, log_hybrid_energy)
            ax.plot(filtered_data_energy, np.exp(intercept + slope * np.log(filtered_data_energy)),
                    color=color_map[id])

            # Calculate RMSE for each car
            mape_before = calculate_mape(np.array(energies_data[id]), np.array(energies_phys[id]))
            mape_after = calculate_mape(np.array(energies_data[id]), np.array(energies_hybrid[id]))

            ax.text(0.05, 0.95 - j * 0.08,
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
        # ax.set_xlim(1, None)
        # ax.set_ylim(1, None)

        # Set x and y limits using global min and max
        ax.set_xlim(1, global_y_max * 1.05)
        ax.set_ylim(1, global_y_max * 1.05)

        # Add legend for C and D
        ax.legend(loc='upper right')
        # Set titles, labels, and markers for C and D
        ax.set_xlabel('Data Energy (kWh)')
        ax.set_ylabel('Model Energy (kWh)')
        ax.text(-0.1, 1.05, chr(67 + i), transform=ax.transAxes, size=16, weight='bold')
        ax.set_title(f"{selected_car}'s Driver : Data Energy vs. Hybrid Model Energy")

    save_path = os.path.join(fig_save_path, 'figureS6.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

figure5(vehicle_files, selected_cars)