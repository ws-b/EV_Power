import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import plotly.graph_objects as go
from GS_Functions import get_vehicle_files, compute_rrmse, compute_rmse, compute_mape, calculate_rmse, calculate_mape, calculate_rrmse
from scipy.interpolate import griddata
from scipy.stats import linregress
from tqdm import tqdm
from GS_vehicle_dict import vehicle_dict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Function to get file lists for each vehicle based on vehicle_dict
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


# def figure3(img1_path, img2_path, save_path, figsize=(6, 10), dpi=300):
#
#     # Load the two images
#     img1 = mpimg.imread(img1_path)
#     img2 = mpimg.imread(img2_path)
#
#     # Create a figure with two subplots
#     fig, axs = plt.subplots(2, 1, figsize=figsize)  # Adjust figure size as needed
#
#     # Display the images in the subplots
#     axs[0].imshow(img1)
#     axs[1].imshow(img2)
#
#     # Hide axes for both subplots
#     for ax in axs:
#         ax.axis('off')
#
#     # Add 'A' and 'B' labels to the top-left corner of each subplot
#     axs[0].text(-0.1, 1.1, 'A', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
#     axs[1].text(-0.1, 1.1, 'B', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
#
#     # Save the final image
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=dpi)
#     plt.show()
#
# img1_path = r"C:\Users\BSL\Desktop\Figures\RRMSE\EV6_RRMSE.png"
# img2_path = r"C:\Users\BSL\Desktop\Figures\RRMSE\Ioniq5_RRMSE.png"
# save_path = r'C:\Users\BSL\Desktop\figure3.png'

# Example usage of the function
directory = r"D:\SamsungSTF\Processed_Data\TripByTrip"
vehicle_files = get_file_lists(directory)
selected_cars = ['EV6', 'Ioniq5']

city_cycle1 = r"D:\SamsungSTF\Processed_Data\TripByTrip\bms_01241228132-2023-06-trip-67.csv"
highway_cycle1 = r"D:\SamsungSTF\Processed_Data\TripByTrip\bms_01241228094-2023-11-trip-88.csv"
city_cycle2 = r"D:\SamsungSTF\Processed_Data\TripByTrip\bms_01241228003-2023-08-trip-11.csv"
highway_cycle2 = r"D:\SamsungSTF\Processed_Data\TripByTrip\bms_01241228107-2023-01-trip-15.csv"
#save_path
fig_save_path = r"C:\Users\BSL\Desktop\Figures"

def figure1(file_lists_ev6, file_lists_ioniq5):
    # Official fuel efficiency data (km/kWh)
    official_efficiency = {
        'Ioniq5': [4.148, 6.221],
        'EV6': [4.085, 6.407]
    }

    # Function to process energy data
    def process_energy_data(file_lists):
        dis_energies_data = []
        for file in tqdm(file_lists):
            data = pd.read_csv(file)

            t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            t_diff = t.diff().dt.total_seconds().fillna(0)
            t_diff = np.array(t_diff.fillna(0))

            v = data['speed']
            v = np.array(v)

            distance = v * t_diff
            total_distance = distance.cumsum()

            Power_data = np.array(data['Power_data'])
            energy_data = Power_data * t_diff / 3600 / 1000

            dis_data_energy = ((total_distance[-1] / 1000) / (energy_data.cumsum()[-1])) if energy_data.cumsum()[
                                                                                              -1] != 0 else 0
            dis_energies_data.append(dis_data_energy)

        return dis_energies_data

    # Function to add official efficiency range for a specific car
    def add_efficiency_lines(selected_car):
        if selected_car in official_efficiency:
            eff_range = official_efficiency[selected_car]
            if len(eff_range) == 2:
                ylim = plt.gca().get_ylim()
                plt.fill_betweenx(ylim, eff_range[0], eff_range[1], color='orange', alpha=0.3, hatch='/')
                plt.text(eff_range[1] + 0.15, plt.gca().get_ylim()[1] * 0.8, 'EPA Efficiency',
                         color='orange', fontsize=12, alpha=0.7)

    # Process the data for EV6 and Ioniq5
    dis_energies_ev6 = process_energy_data(file_lists_ev6)
    dis_energies_ioniq5 = process_energy_data(file_lists_ioniq5)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    # Plot for EV6
    plt.sca(ax1)  # Set current axis to ax1
    mean_value_ev6 = np.mean(dis_energies_ev6)
    sns.histplot(dis_energies_ev6, bins='auto', color='gray', kde=False)
    plt.axvline(mean_value_ev6, color='red', linestyle='--')
    plt.text(mean_value_ev6 + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value_ev6:.2f}', color='red', fontsize=12, alpha=0.7)
    plt.xlabel('Efficiency in km/kWh')
    plt.xlim((0, 15))
    plt.ylim(0, 1300)
    plt.ylabel('Number of trips')
    ax1.text(-0.1, 1.05, "A", transform=ax1.transAxes, size=14, weight='bold', ha='left')  # Move (a) to top-left
    ax1.set_title("Energy Consumption Distribution : EV6", pad=10)  # Title below (a)
    add_efficiency_lines('EV6')
    plt.grid(False)

    # Plot for Ioniq5
    plt.sca(ax2)  # Set current axis to ax2
    mean_value_ioniq5 = np.mean(dis_energies_ioniq5)
    sns.histplot(dis_energies_ioniq5, bins='auto', color='gray', kde=False)
    plt.axvline(mean_value_ioniq5, color='red', linestyle='--')
    plt.text(mean_value_ioniq5 + 0.05, plt.gca().get_ylim()[1] * 0.95, f'Mean: {mean_value_ioniq5:.2f}', color='red', fontsize=12, alpha=0.7)
    plt.xlabel('Efficiency in km/kWh')
    plt.xlim(0, 15)
    plt.ylim(0, 900)
    plt.ylabel('Number of trips')
    ax2.text(-0.1, 1.05, "B", transform=ax2.transAxes, size=14, weight='bold', ha='left')  # Move (b) to top-left
    ax2.set_title("Energy Consumption Distribution : Ioniq5", pad=10)  # Title below (b)
    add_efficiency_lines('Ioniq5')
    plt.grid(False)

    # Save the figure with dpi 300
    save_path = os.path.join(fig_save_path, 'figure1.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def figure4(city_cycle1, highway_cycle1, city_cycle2, highway_cycle2):
    fig, axs = plt.subplots(2, 4, figsize=(24, 10))

    def process_and_plot_power(file, ax, marker, title):
        data = pd.read_csv(file)

        # Time processing
        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff)
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60
        date = t.iloc[0].strftime('%Y-%m-%d')

        # Power data
        power_data = np.array(data['Power_data']) / 1000  # Convert to kW
        power_phys = np.array(data['Power_phys']) / 1000  # Convert to kW

        # Hybrid power (if available)
        if 'Power_hybrid' in data.columns:
            power_hybrid = np.array(data['Power_hybrid']) / 1000

        # Plot the comparison graph
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Power (kW)')
        ax.plot(t_min, power_data, label='Data Power (kW)', color='tab:blue', alpha=0.6)
        ax.plot(t_min, power_phys, label='Physics Model Power (kW)', color='tab:red', alpha=0.6)

        if 'Power_hybrid' in data.columns:
            ax.plot(t_min, power_hybrid, label='Hybrid Model Power (kW)', color='tab:green', alpha=0.6)

        ax.set_ylim()

        ax.legend(loc='upper left', bbox_to_anchor=(0, 0.99))
        ax.set_title(title, pad=10)
        ax.text(-0.1, 1.05, marker, transform=ax.transAxes, size=14, weight='bold', ha='left')  # Add marker

    def process_and_plot_energy(file, ax, marker, title):
        data = pd.read_csv(file)

        # Time processing
        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff)
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

        # Plot the comparison graph
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('BMS Energy and Physics Model Energy (kWh)')
        ax.plot(t_min, energy_data_cumulative, label='Data Energy (kWh)', color='tab:blue', alpha=0.6)
        ax.plot(t_min, energy_phys_cumulative, label='Physics Model Energy (kWh)', color='tab:red', alpha=0.6)
        if 'Power_hybrid' in data.columns:
            ax.plot(t_min, energy_hybrid_cumulative, label='Hybrid Model Energy (kWh)', color='tab:green', alpha=0.6)

        ax.legend(loc='upper left', bbox_to_anchor=(0, 0.99))
        ax.set_title(title, pad=10)
        ax.text(-0.1, 1.05, marker, transform=ax.transAxes, size=14, weight='bold', ha='left')  # Add marker

    # Plot for city_cycle1 power in the first row, first column
    process_and_plot_power(city_cycle1, axs[0, 0], 'A', 'City Cycle 1 - Power Comparison')

    # Plot for highway_cycle1 power in the second row, first column
    process_and_plot_power(highway_cycle1, axs[0, 1], 'C', 'Highway Cycle 1 - Power Comparison')

    # Plot for city_energy_file in the first row, second column
    process_and_plot_energy(city_cycle1, axs[1, 0], 'B', 'City Cycle 1 - Energy Comparison')

    # Plot for highway_energy_file in the second row, second column
    process_and_plot_energy(highway_cycle1, axs[1, 1], 'D', 'Highway Cycle 1 - Energy Comparison')

    # Plot for city_cycle1 power in the first row, first column
    process_and_plot_power(city_cycle2, axs[0, 2], 'E', 'City Cycle 2 - Power Comparison')

    # Plot for highway_cycle1 power in the second row, first column
    process_and_plot_power(highway_cycle2, axs[0, 3], 'G', 'Highway Cycle 2 - Power Comparison')

    # Plot for city_energy_file in the first row, second column
    process_and_plot_energy(city_cycle2, axs[1, 2], 'F', 'City Cycle 2 - Energy Comparison')

    # Plot for highway_energy_file in the second row, second column
    process_and_plot_energy(highway_cycle2, axs[1, 3], 'H', 'Highway Cycle 2 - Energy Comparison')

    # Adjust layout and save the figure
    plt.tight_layout()
    save_path = os.path.join(fig_save_path, 'figure4.png')  # Replace with your save path
    plt.savefig(save_path, dpi=300)
    plt.show()

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
        relative_rmse_before = calculate_rrmse(np.array(energies_dict[selected_car]['data']),
                                               np.array(energies_dict[selected_car]['phys']))
        mape_after = calculate_mape(np.array(energies_dict[selected_car]['data']),
                                    np.array(energies_dict[selected_car]['hybrid']))
        relative_rmse_after = calculate_rrmse(np.array(energies_dict[selected_car]['data']),
                                              np.array(energies_dict[selected_car]['hybrid']))

        # Displaying the MAPE and RRMSE values in the plot
        ax.text(0.6, 0.15,
                f'MAPE (Before): {mape_before:.2f}%\n'
                f'RRMSE (Before): {relative_rmse_before:.2%}\n'
                f'MAPE (After): {mape_after:.2f}%\n'
                f'RRMSE (After): {relative_rmse_after:.2%}',
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

        colors = cm.rainbow(np.linspace(0, 1, len(sample_files_dict)))
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
            mape_before, relative_rmse_before = calculate_mape(np.array(energies_data[id]),
                                                               np.array(energies_phys[id])), calculate_rrmse(
                np.array(energies_data[id]), np.array(energies_phys[id]))
            mape_after, relative_rmse_after = calculate_mape(np.array(energies_data[id]),
                                                             np.array(energies_hybrid[id])), calculate_rrmse(
                np.array(energies_data[id]), np.array(energies_hybrid[id]))

            ax.text(0.05, 0.95 - j * 0.13,
                    f'{selected_car} {driver_label}\nMAPE (Before): {mape_before:.2f}%\nRRMSE (Before): {relative_rmse_before:.2%}\nMAPE (After): {mape_after:.2f}%\nRRMSE (After): {relative_rmse_after:.2%}',
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
        ax.text(-0.1, 1.05, chr(67 + i), transform=ax.transAxes, size=14, weight='bold')
        ax.set_title(f"{selected_car}'s Driver : Data Energy vs. Hybrid Model Energy")

    save_path = os.path.join(fig_save_path, 'figure5.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def figure6(file_lists_ev6, file_lists_ioniq5):
    # Official fuel efficiency data (km/kWh)
    official_efficiency = {
        'Ioniq5': [4.148, 6.221],
        'EV6': [4.085, 6.407]
    }

    # Function to process energy data
    def process_energy_data(file_lists):
        dis_energies_phys = []
        dis_energies_data = []
        dis_energies_hybrid = []
        for file in tqdm(file_lists):
            data = pd.read_csv(file)

            t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            t_diff = t.diff().dt.total_seconds().fillna(0)
            t_diff = np.array(t_diff.fillna(0))

            v = data['speed']
            v = np.array(v)

            distance = v * t_diff
            total_distance = distance.cumsum()

            Power_data = np.array(data['Power_data'])
            energy_data = Power_data * t_diff / 3600 / 1000
            power_phys = np.array(data['Power_phys'])
            energy_phys = power_phys * t_diff / 3600 / 1000
            power_hybrid = np.array(data['Power_hybrid'])
            energy_hybrid = power_hybrid * t_diff / 3600 / 1000

            dis_data_energy = ((total_distance[-1] / 1000) / (energy_data.cumsum()[-1])) if energy_data.cumsum()[
                                                                                                -1] != 0 else 0
            dis_energies_data.append(dis_data_energy)
            dis_energy_phys = ((total_distance[-1] / 1000) / (energy_phys.cumsum()[-1])) if energy_phys.cumsum()[
                                                                                                -1] != 0 else 0
            dis_energies_phys.append(dis_energy_phys)
            dis_energy_hybrid = ((total_distance[-1] / 1000) / (energy_hybrid.cumsum()[-1])) if energy_hybrid.cumsum()[
                                                                                                    -1] != 0 else 0
            dis_energies_hybrid.append(dis_energy_hybrid)

        return dis_energies_phys, dis_energies_data, dis_energies_hybrid

    # Function to add official efficiency range for a specific car
    def add_efficiency_lines(selected_car):
        if selected_car in official_efficiency:
            eff_range = official_efficiency[selected_car]
            if len(eff_range) == 2:
                ylim = plt.gca().get_ylim()
                plt.fill_betweenx(ylim, eff_range[0], eff_range[1], color='orange', alpha=0.3, hatch='/')
                plt.text(eff_range[1] + 0.15, plt.gca().get_ylim()[1] * 0.8, 'EPA Efficiency',
                         color='orange', fontsize=12, alpha=0.7)

    # Process the data for EV6 and Ioniq5
    dis_energies_ev6 = process_energy_data(file_lists_ev6)
    dis_energies_ioniq5 = process_energy_data(file_lists_ioniq5)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    # Plot for EV6
    plt.sca(ax1)  # Set current axis to ax1
    mean_value_ev6_phys = np.mean(dis_energies_ev6[0])
    mean_value_ev6_data = np.mean(dis_energies_ev6[1])
    mean_value_ev6_hybrid = np.mean(dis_energies_ev6[2])

    sns.histplot(dis_energies_ev6[1], bins='auto', color='gray', kde=False, label='Data', alpha=0.5)
    sns.histplot(dis_energies_ev6[0], bins='auto', color='blue', kde=False, label='Physics-based Model', alpha=0.5)
    sns.histplot(dis_energies_ev6[2], bins='auto', color='green', kde=False, label='Hybrid Model', alpha=0.5)

    plt.axvline(mean_value_ev6_phys, color='blue', linestyle='--')
    plt.axvline(mean_value_ev6_data, color='gray', linestyle='--')
    plt.axvline(mean_value_ev6_hybrid, color='green', linestyle='--')

    plt.text(mean_value_ev6_data + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value_ev6_data:.2f}',
             color='gray', fontsize=12, alpha=0.7)
    plt.text(mean_value_ev6_hybrid + 0.05, plt.gca().get_ylim()[1] * 0.65, f'Mean: {mean_value_ev6_hybrid:.2f}',
             color='green', fontsize=12, alpha=0.7)
    plt.xlabel('Efficiency in km/kWh')
    plt.xlim((0, 15))
    plt.ylim(0, 1300)
    plt.ylabel('Number of trips')
    ax1.text(-0.1, 1.05, "A", transform=ax1.transAxes, size=14, weight='bold', ha='left')  # Move (a) to top-left
    ax1.set_title("Energy Consumption Distribution : EV6", pad=10)  # Title below (a)
    add_efficiency_lines('EV6')
    plt.grid(False)
    plt.legend()

    # Plot for Ioniq5
    plt.sca(ax2)  # Set current axis to ax2
    mean_value_ioniq5_phys = np.mean(dis_energies_ioniq5[0])
    mean_value_ioniq5_data = np.mean(dis_energies_ioniq5[1])
    mean_value_ioniq5_hybrid = np.mean(dis_energies_ioniq5[2])

    sns.histplot(dis_energies_ioniq5[1], bins='auto', color='gray', kde=False, label='Data', alpha=0.5)
    sns.histplot(dis_energies_ioniq5[0], bins='auto', color='blue', kde=False, label='Physics-based Model', alpha=0.5)
    sns.histplot(dis_energies_ioniq5[2], bins='auto', color='green', kde=False, label='Hybrid Model', alpha=0.5)

    plt.axvline(mean_value_ioniq5_phys, color='blue', linestyle='--')
    plt.axvline(mean_value_ioniq5_data, color='gray', linestyle='--')
    plt.axvline(mean_value_ioniq5_hybrid, color='green', linestyle='--')

    plt.text(mean_value_ioniq5_data + 0.05, plt.gca().get_ylim()[1] * 0.95, f'Mean: {mean_value_ioniq5_data:.2f}',
             color='gray', fontsize=12, alpha=0.7)
    plt.text(mean_value_ioniq5_hybrid + 0.05, plt.gca().get_ylim()[1] * 0.65, f'Mean: {mean_value_ioniq5_hybrid:.2f}',
             color='green', fontsize=12, alpha=0.7)
    plt.xlabel('Efficiency in km/kWh')
    plt.xlim(0, 15)
    plt.ylim(0, 900)
    plt.ylabel('Number of trips')
    ax2.text(-0.1, 1.05, "B", transform=ax2.transAxes, size=14, weight='bold', ha='left')  # Move (b) to top-left
    ax2.set_title("Energy Consumption Distribution : Ioniq5", pad=10)  # Title below (b)
    add_efficiency_lines('Ioniq5')
    plt.grid(False)
    plt.legend()

    # Save the figure with dpi 300
    save_path = os.path.join(fig_save_path, 'figure6.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# figure1(vehicle_files['EV6'], vehicle_files['Ioniq5'])
# figure3(img1_path, img2_path, save_path)
# figure4(city_cycle1, highway_cycle1, city_cycle2, highway_cycle2)
figure5(vehicle_files, selected_cars)
# figure6(vehicle_files['EV6'], vehicle_files['Ioniq5'])