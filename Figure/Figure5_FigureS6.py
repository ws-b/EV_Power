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
    n = len(x)
    mean_x = np.mean(x)
    Sxx = np.sum((x - mean_x) ** 2)

    y_pred = intercept + slope * x
    residuals = y - y_pred
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))

    t_val = t.ppf((1 + confidence) / 2., n - 2)

    y_pred_new = intercept + slope * x_val
    se_new = s_err * np.sqrt(1 + (x_val - mean_x)**2 / Sxx)
    y_lower_new = y_pred_new - t_val * se_new
    y_upper_new = y_pred_new + t_val * se_new

    return y_pred_new, y_lower_new, y_upper_new

def figure5(vehicle_files, selected_cars):
    # 1) 모든 데이터 로드
    all_energies_dict = {car: {'data': [], 'phys': [], 'hybrid': []} for car in selected_cars}
    global_x_max = float('-inf')
    global_y_max = float('-inf')

    for selected_car in selected_cars:
        for file in tqdm(vehicle_files[selected_car]):
            data = pd.read_csv(file)

            t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            t_diff = t.diff().dt.total_seconds().fillna(0).values

            power_data = data['Power_data'].values
            energy_data = power_data * t_diff / 3600 / 1000
            all_energies_dict[selected_car]['data'].append(energy_data.cumsum()[-1])

            if 'Power_phys' in data.columns:
                power_phys = data['Power_phys'].values
                energy_phys = power_phys * t_diff / 3600 / 1000
                all_energies_dict[selected_car]['phys'].append(energy_phys.cumsum()[-1])

            if 'Power_hybrid' in data.columns:
                power_hybrid = data['Power_hybrid'].values
                energy_hybrid = power_hybrid * t_diff / 3600 / 1000
                all_energies_dict[selected_car]['hybrid'].append(energy_hybrid.cumsum()[-1])

    # 2) 전역 최댓값 구하기
    for car in selected_cars:
        data_arr = np.array(all_energies_dict[car]['data'])
        phys_arr = np.array(all_energies_dict[car]['phys'])
        hybrid_arr = np.array(all_energies_dict[car]['hybrid'])

        global_x_max = max(global_x_max, data_arr.max())
        global_y_max = max(global_y_max, phys_arr.max(), hybrid_arr.max(), global_x_max)

    # 3) 2x2 Subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # ------------------- (위쪽) 1행 A, B -------------------
    for i, selected_car in enumerate(selected_cars):
        ax = axs[0, i]

        ax.set_xlabel('Data Energy (kWh)')
        ax.set_ylabel('Model Energy (kWh)')
        ax.text(-0.1, 1.05, chr(65 + i), transform=ax.transAxes, size=16, weight='bold')  # A or B

        data_list = np.array(all_energies_dict[selected_car]['data'])
        phys_list = np.array(all_energies_dict[selected_car]['phys'])
        hybrid_list = np.array(all_energies_dict[selected_car]['hybrid'])

        # -------------------- [로그 스케일 회귀분석 선] --------------------
        log_data_energy = np.log(data_list)
        log_phys_energy = np.log(phys_list)
        log_hybrid_energy = np.log(hybrid_list)

        # (1) Physics-based 회귀
        slope_phys, intercept_phys, _, _, _ = linregress(log_data_energy, log_phys_energy)
        # (2) Hybrid(XGB) 회귀
        slope_hyb, intercept_hyb, _, _, _ = linregress(log_data_energy, log_hybrid_energy)

        x_min = data_list.min()
        x_max = data_list.max()
        x_vals = np.linspace(x_min, x_max, 100)

        # 회귀선
        y_phys = np.exp(intercept_phys) * x_vals**slope_phys
        y_hybrid = np.exp(intercept_hyb) * x_vals**slope_hyb

        line_phys, = ax.plot(x_vals, y_phys, color="#efc000ff", label='Physics-based Model')
        line_hybrid, = ax.plot(x_vals, y_hybrid, color="#cd534cff", label='Hybrid Model(XGB)')

        # 예측 구간(Physics)
        _, y_phys_lower, y_phys_upper = calculate_prediction_interval(
            np.log(x_vals), log_data_energy, log_phys_energy, slope_phys, intercept_phys
        )
        ax.fill_between(x_vals, np.exp(y_phys_lower), np.exp(y_phys_upper),
                        color="#efc000ff", alpha=0.2, label='_nolegend_')

        # 예측 구간(Hybrid)
        _, y_hyb_lower, y_hyb_upper = calculate_prediction_interval(
            np.log(x_vals), log_data_energy, log_hybrid_energy, slope_hyb, intercept_hyb
        )
        ax.fill_between(x_vals, np.exp(y_hyb_lower), np.exp(y_hyb_upper),
                        color="#cd534cff", alpha=0.2, label='_nolegend_')
        # --------------------------------------------------------------

        # -------------------- [동일 샘플 스캐터] --------------------
        # 3개 모델 모두 0보다 큰 값을 만족하는 지점만 교집합으로 취급
        mask_both = (data_list > 0) & (phys_list > 0) & (hybrid_list > 0)

        data_both = data_list[mask_both]
        phys_both = phys_list[mask_both]
        hybr_both = hybrid_list[mask_both]

        # 최대 1000개까지 샘플링
        if len(data_both) > 1000:
            indices = random.sample(range(len(data_both)), 1000)
            data_both = data_both[indices]
            phys_both = phys_both[indices]
            hybr_both = hybr_both[indices]

        # 물리모델 스캐터
        ax.scatter(
            data_both, phys_both,
            facecolors='none', edgecolors="#efc000ff", alpha=0.6, label='_nolegend_'
        )
        # 하이브리드 스캐터
        ax.scatter(
            data_both, hybr_both,
            facecolors='none', edgecolors="#cd534cff", alpha=0.6, label='_nolegend_'
        )
        # --------------------------------------------------------------

        # 회색 패치(레전드용)
        prediction_patch = mpatches.Patch(color="#747678ff", alpha=0.2, label='Prediction Interval 90%')

        # MAPE 계산
        mape_before = calculate_mape(data_list, phys_list)
        mape_after = calculate_mape(data_list, hybrid_list)

        # MAPE 텍스트
        ax.text(
            0.6, 0.10,
            f'MAPE (Before): {mape_before:.2f}%\n'
            f'MAPE (After): {mape_after:.2f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top'
        )

        # 로그 스케일 & 축 범위 설정
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_aspect('equal')
        ax.set_xlim(1, global_y_max * 1.05)
        ax.set_ylim(1, global_y_max * 1.05)

        # 레전드
        handles = [line_phys, line_hybrid, prediction_patch]
        labels = [h.get_label() for h in handles]
        ax.legend(handles=handles, labels=labels, loc='upper left')

        ax.set_title(f"{selected_car} : Data Energy vs. Hybrid Model Energy")

    # ------------------- (아래쪽) 2행 C, D -------------------
    for i, selected_car in enumerate(selected_cars):
        sample_ids = vehicle_dict[selected_car][0:3]
        sample_files_dict = {id_: [f for f in vehicle_files[selected_car] if id_ in f] for id_ in sample_ids}

        ax = axs[1, i]

        energies_data = {}
        energies_phys = {}
        energies_hybrid = {}

        colors = [
            "#20854eff", "#4dbbd5ff", "#ee4c97ff", "#7e6148ff", "#747678ff"
        ]
        color_map = {}

        for j, (id_, files) in enumerate(sample_files_dict.items()):
            energies_data[id_] = []
            energies_phys[id_] = []
            energies_hybrid[id_] = []
            color_map[id_] = colors[j]
            driver_label = f"Driver {j + 1}"

            for file in tqdm(files, desc=f'Processing {selected_car} - Driver {id_}'):
                data = pd.read_csv(file)

                t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
                t_diff = t.diff().dt.total_seconds().fillna(0).values

                power_data = data['Power_data'].values
                energy_data = power_data * t_diff / 3600 / 1000
                energies_data[id_].append(energy_data.cumsum()[-1])

                if 'Power_phys' in data.columns:
                    power_phys = data['Power_phys'].values
                    energy_phys = power_phys * t_diff / 3600 / 1000
                    energies_phys[id_].append(energy_phys.cumsum()[-1])

                if 'Power_hybrid' in data.columns:
                    power_hybrid = data['Power_hybrid'].values
                    predicted_energy = power_hybrid * t_diff / 3600 / 1000
                    energies_hybrid[id_].append(predicted_energy.cumsum()[-1])

            filtered_data_energy = np.array(energies_data[id_])
            filtered_phys_energy = np.array(energies_phys[id_])
            filtered_hybrid_energy = np.array(energies_hybrid[id_])

            # 로그 스케일을 위해 양수만 필터링
            positive_mask = (filtered_data_energy > 0) & (filtered_hybrid_energy > 0)
            filtered_data_energy = filtered_data_energy[positive_mask]
            filtered_hybrid_energy = filtered_hybrid_energy[positive_mask]

            # 산포도
            ax.scatter(
                filtered_data_energy,
                filtered_hybrid_energy,
                facecolors='none',
                edgecolors=color_map[id_],
                label=f'{driver_label} Hybrid Model(XGB)'
            )

            # 회귀선 (로그 스케일)
            if len(filtered_data_energy) > 0:
                log_data_energy = np.log(filtered_data_energy)
                log_hybrid_energy = np.log(filtered_hybrid_energy)

                slope_, intercept_, _, _, _ = linregress(log_data_energy, log_hybrid_energy)
                ax.plot(
                    filtered_data_energy,
                    np.exp(intercept_ + slope_ * np.log(filtered_data_energy)),
                    color=color_map[id_]
                )

            # MAPE
            mape_before = calculate_mape(np.array(energies_data[id_]), np.array(energies_phys[id_]))
            mape_after = calculate_mape(np.array(energies_data[id_]), np.array(energies_hybrid[id_]))

            ax.text(
                0.05, 0.95 - j * 0.08,
                f'{selected_car} {driver_label}\n'
                f'MAPE (Before): {mape_before:.2f}%\n'
                f'MAPE (After): {mape_after:.2f}%',
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                color=color_map[id_]
            )

        # 대각선 비교선
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_aspect('equal')
        ax.set_xlim(1, global_y_max * 1.05)
        ax.set_ylim(1, global_y_max * 1.05)

        ax.legend(loc='upper right')
        ax.set_xlabel('Data Energy (kWh)')
        ax.set_ylabel('Model Energy (kWh)')
        ax.text(-0.1, 1.05, chr(67 + i), transform=ax.transAxes, size=16, weight='bold')  # C or D
        ax.set_title(f"{selected_car}'s Driver : Data Energy vs. Hybrid Model Energy")

    save_path = os.path.join(fig_save_path, 'figureS6.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# 실행 예시
figure5(vehicle_files, selected_cars)
