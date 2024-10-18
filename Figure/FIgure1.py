import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from GS_vehicle_dict import vehicle_dict
import matplotlib.pyplot as plt

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

def figure1(file_lists_ev6, file_lists_ioniq5):
    # Official fuel efficiency data (km/kWh)
    official_efficiency = {
        'Ioniq5': [4.202, 6.303],
        'EV6': [4.106, 6.494]
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
    plt.ylim(0, 1400)
    plt.ylabel('Number of trips')
    ax1.text(-0.1, 1.05, "A", transform=ax1.transAxes, size=16, weight='bold', ha='left')  # Move (a) to top-left
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
    ax2.text(-0.1, 1.05, "B", transform=ax2.transAxes, size=16, weight='bold', ha='left')  # Move (b) to top-left
    ax2.set_title("Energy Consumption Distribution : Ioniq5", pad=10)  # Title below (b)
    add_efficiency_lines('Ioniq5')
    plt.grid(False)

    # Save the figure with dpi 300
    save_path = os.path.join(fig_save_path, 'figure1.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

figure1(vehicle_files['EV6'], vehicle_files['Ioniq5'])