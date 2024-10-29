import os
import numpy as np
import pandas as pd
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

city_cycle1 = r"D:\SamsungSTF\Processed_Data\TripByTrip\bms_01241228132-2023-06-trip-67.csv"
highway_cycle1 = r"D:\SamsungSTF\Processed_Data\TripByTrip\bms_01241228094-2023-11-trip-88.csv"
city_cycle2 = r"D:\SamsungSTF\Processed_Data\TripByTrip\bms_01241228003-2023-08-trip-11.csv"
highway_cycle2 = r"D:\SamsungSTF\Processed_Data\TripByTrip\bms_01241228107-2023-01-trip-15.csv"

#save_path
fig_save_path = r"C:\Users\BSL\Desktop\Figures"

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
        ax.text(-0.1, 1.05, marker, transform=ax.transAxes, size=16, weight='bold', ha='left')  # Add marker

    # Plot for city_cycle1 power in the first row, first column
    process_and_plot_power(city_cycle1, axs[0, 0], 'A', 'City Cycle 1 - Power Comparison')

    # Plot for city_cycle1 energy in the second row, first column
    process_and_plot_energy(city_cycle1, axs[1, 0], 'B', 'City Cycle 1 - Energy Comparison')

    # Plot for city_cycle2 power in the first row, second column
    process_and_plot_power(city_cycle2, axs[0, 1], 'C', 'City Cycle 2 - Power Comparison')

    # Plot for city_cycle2 energy in the second row, second column
    process_and_plot_energy(city_cycle2, axs[1, 1], 'D', 'City Cycle 2 - Energy Comparison')

    # Plot for highway_cycle1 power in the first row, third column
    process_and_plot_power(highway_cycle1, axs[0, 2], 'E', 'Highway Cycle 1 - Power Comparison')

    # Plot for highway_cycle1 energy in the second row, third column
    process_and_plot_energy(highway_cycle1, axs[1, 2], 'F', 'Highway Cycle 1 - Energy Comparison')

    # Plot for highway_cycle2 power in the first row, fourth column
    process_and_plot_power(highway_cycle2, axs[0, 3], 'G', 'Highway Cycle 2 - Power Comparison')

    # Plot for highway_cycle2 energy in the second row, fourth column
    process_and_plot_energy(highway_cycle2, axs[1, 3], 'H', 'Highway Cycle 2 - Energy Comparison')

    # Adjust layout and save the figure
    plt.tight_layout()
    save_path = os.path.join(fig_save_path, 'figure4.png')  # Replace with your save path
    plt.savefig(save_path, dpi=300)
    plt.show()

figure4(city_cycle1, highway_cycle1, city_cycle2, highway_cycle2)