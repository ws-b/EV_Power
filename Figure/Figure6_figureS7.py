import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from GS_vehicle_dict import vehicle_dict
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

def figure6(file_lists_ev6, file_lists_ioniq5):
    # Official fuel efficiency data (km/kWh)
    official_efficiency = {
        'Ioniq5': [158.7, 238.0],
        'EV6': [154.0, 243.5],
        'KonaEV': [159.9 ,203.3],
        'NiroEV': [166.2, 207.4],
        'GV60': [167.5, 255.4],
        'Ioniq6': [136.9, 222.8]
    }

    ylim = {
        'KonaEV': [0, 700],
        'NiroEV': [0, 500],
        'GV60': [0, 250],
        'Ioniq6': [0, 160]
    }

    # Set font sizes using the scaling factor
    scaling = 1
    plt.rcParams['font.size'] = 10 * scaling  # Base font size
    plt.rcParams['axes.titlesize'] = 12 * scaling  # Title font size
    plt.rcParams['axes.labelsize'] = 10 * scaling  # Axis label font size
    plt.rcParams['xtick.labelsize'] = 10 * scaling  # X-axis tick label font size
    plt.rcParams['ytick.labelsize'] = 10 * scaling  # Y-axis tick label font size
    plt.rcParams['legend.fontsize'] = 10 * scaling  # Legend font size
    plt.rcParams['legend.title_fontsize'] = 10 * scaling  # Legend title font size
    plt.rcParams['figure.titlesize'] = 12 * scaling  # Figure title font size

    # Function to process energy data
    def process_energy_data(file_lists):
        all_ecr_phys = []
        all_ecr_data = []
        all_ecr_hybrid = []
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
            energy_data = Power_data * t_diff / 3600
            power_phys = np.array(data['Power_phys'])
            energy_phys = power_phys * t_diff / 3600
            power_hybrid = np.array(data['Power_hybrid'])
            energy_hybrid = power_hybrid * t_diff / 3600

            ecr_phys = ((energy_phys.cumsum()[-1])/ (total_distance[-1] / 1000)) if energy_data.cumsum()[
                                                                                              -1] != 0 else 0
            all_ecr_phys.append(ecr_phys)

            ecr_data = ((energy_data.cumsum()[-1])/ (total_distance[-1] / 1000)) if energy_phys.cumsum()[
                                                                                              -1] != 0 else 0
            all_ecr_data.append(ecr_data)

            ecr_hybrid = ((energy_hybrid.cumsum()[-1])/ (total_distance[-1] / 1000)) if energy_hybrid.cumsum()[
                                                                                              -1] != 0 else 0
            all_ecr_hybrid.append(ecr_hybrid)

        return all_ecr_phys, all_ecr_data, all_ecr_hybrid

    # Function to add official efficiency range for a specific car
    def add_efficiency_lines(selected_car):
        if selected_car in official_efficiency:
            eff_range = official_efficiency[selected_car]
            if len(eff_range) == 2:
                ylim = plt.gca().get_ylim()
                plt.fill_betweenx(ylim, eff_range[0], eff_range[1], color="#E18727FF", alpha=0.3, hatch='/')
                plt.text(eff_range[1] + 0.15, plt.gca().get_ylim()[1] * 0.6, 'EPA Efficiency',
                         color="#E18727FF", fontsize=12, alpha=0.9)

    # Process the data for EV6 and Ioniq5
    dis_energies_ev6 = process_energy_data(file_lists_ev6)
    dis_energies_ioniq5 = process_energy_data(file_lists_ioniq5)

    # Define common bins
    bin_start = 50
    bin_end = 400
    num_bins = 70  # Adjust the number of bins as needed
    bins = np.linspace(bin_start, bin_end, num_bins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    label = ["A", "B"] if selected_cars[0] == "KonaEV" else ["C", "D"]
    # Plot for EV6
    plt.sca(ax1)  # Set current axis to ax1
    mean_value_ev6_phys = np.mean(dis_energies_ev6[0])
    mean_value_ev6_data = np.mean(dis_energies_ev6[1])
    mean_value_ev6_hybrid = np.mean(dis_energies_ev6[2])

    sns.histplot(dis_energies_ev6[1], bins=bins, color="#747678FF", kde=False, label='Data', alpha=0.5)
    sns.histplot(dis_energies_ev6[0], bins=bins, color="#0073C2FF", kde=False, label='Physics-based Model', alpha=0.5)
    sns.histplot(dis_energies_ev6[2], bins=bins, color="#EFC000FF", kde=False, label='Hybrid Model(XGB)', alpha=0.5)

    plt.axvline(mean_value_ev6_phys, color="#0073C2FF", linestyle='--')
    plt.axvline(mean_value_ev6_data, color="#747678FF", linestyle='--')
    plt.axvline(mean_value_ev6_hybrid, color="#EFC000FF", linestyle='--')

    plt.text(mean_value_ev6_data + 0.05, plt.gca().get_ylim()[1] * 0.77, f'Mean: {mean_value_ev6_data:.2f}',
             color="#747678FF", fontsize=12, alpha=0.7)
    plt.text(mean_value_ev6_hybrid + 0.05, plt.gca().get_ylim()[1] * 0.45, f'Mean: {mean_value_ev6_hybrid:.2f}',
             color="#EFC000FF", fontsize=12, alpha=0.7)
    plt.text(mean_value_ev6_phys + 0.05, plt.gca().get_ylim()[1] * 0.45, f'Mean: {mean_value_ev6_phys:.2f}',
             color="#0073C2FF", fontsize=12, alpha=0.7)
    plt.xlabel('ECR(Wh/km)')
    plt.xlim((50, 400))
    plt.ylim(ylim[selected_cars[0]])
    plt.ylabel('Number of trips')
    ax1.text(-0.1, 1.05, label[0], transform=ax1.transAxes, size=16, weight='bold', ha='left')  # Move (a) to top-left
    ax1.set_title(f"Energy Consumption Rate Distribution : {selected_cars[0]}", pad=10)  # Title below (a)
    add_efficiency_lines(f'{selected_cars[0]}')
    plt.grid(False)
    plt.legend(loc='upper right')

    # Plot for Ioniq5
    plt.sca(ax2)  # Set current axis to ax2
    mean_value_ioniq5_phys = np.mean(dis_energies_ioniq5[0])
    mean_value_ioniq5_data = np.mean(dis_energies_ioniq5[1])
    mean_value_ioniq5_hybrid = np.mean(dis_energies_ioniq5[2])

    sns.histplot(dis_energies_ioniq5[1], bins=bins, color="#747678FF", kde=False, label='Data', alpha=0.5)
    sns.histplot(dis_energies_ioniq5[0], bins=bins, color="#0073C2FF", kde=False, label='Physics-based Model', alpha=0.5)
    sns.histplot(dis_energies_ioniq5[2], bins=bins, color="#EFC000FF", kde=False, label='Hybrid Model(XGB)', alpha=0.5)

    plt.axvline(mean_value_ioniq5_phys, color="#0073C2FF", linestyle='--')
    plt.axvline(mean_value_ioniq5_data, color="#747678FF", linestyle='--')
    plt.axvline(mean_value_ioniq5_hybrid, color="#EFC000FF", linestyle='--')

    plt.text(mean_value_ioniq5_data + 0.05, plt.gca().get_ylim()[1] * 0.77, f'Mean: {mean_value_ioniq5_data:.2f}',
             color="#747678FF", fontsize=12, alpha=0.7)
    plt.text(mean_value_ioniq5_hybrid + 0.05, plt.gca().get_ylim()[1] * 0.45, f'Mean: {mean_value_ioniq5_hybrid:.2f}',
             color="#EFC000FF", fontsize=12, alpha=0.7)
    plt.text(mean_value_ioniq5_phys + 0.05, plt.gca().get_ylim()[1] * 0.45, f'Mean: {mean_value_ioniq5_phys:.2f}',
             color="#0073C2FF", fontsize=12, alpha=0.7)
    plt.xlabel('ECR(Wh/km)')
    plt.xlim(50, 400)
    plt.ylim(ylim[selected_cars[1]])
    plt.ylabel('Number of trips')
    ax2.text(-0.1, 1.05, label[1], transform=ax2.transAxes, size=16, weight='bold', ha='left')  # Move (b) to top-left
    ax2.set_title(f"Energy Consumption Rate Distribution : {selected_cars[1]}", pad=10)  # Title below (b)
    add_efficiency_lines(f'{selected_cars[1]}')
    plt.grid(False)
    plt.legend(loc='upper right')

    # Save the figure with dpi 300
    file_name = 'figureS7.png'
    save_path = os.path.join(fig_save_path, file_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


figure6(vehicle_files[selected_cars[0]], vehicle_files[selected_cars[1]])