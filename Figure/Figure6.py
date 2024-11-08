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
selected_cars = ['EV6', 'Ioniq5']

#save_path
fig_save_path = r"C:\Users\BSL\Desktop\Figures"

def figure6(file_lists_ev6, file_lists_ioniq5):
    # Official fuel efficiency data (km/kWh)
    official_efficiency = {
        'Ioniq5': [4.202, 6.303],
        'EV6': [4.106, 6.494]
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
    plt.ylim(0, 1400)
    plt.ylabel('Number of trips')
    ax1.text(-0.1, 1.05, "A", transform=ax1.transAxes, size=16, weight='bold', ha='left')  # Move (a) to top-left
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
    ax2.text(-0.1, 1.05, "B", transform=ax2.transAxes, size=16, weight='bold', ha='left')  # Move (b) to top-left
    ax2.set_title("Energy Consumption Distribution : Ioniq5", pad=10)  # Title below (b)
    add_efficiency_lines('Ioniq5')
    plt.grid(False)
    plt.legend()

    # Save the figure with dpi 300
    save_path = os.path.join(fig_save_path, 'figure6.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


figure6(vehicle_files['EV6'], vehicle_files['Ioniq5'])