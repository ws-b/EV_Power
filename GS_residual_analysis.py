import os
import platform
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from GS_preprocessing import get_file_list
import numpy as np
import random

# Define vehicle types
vehicle_types = {
    '01241248726': 'kona EV',
    '01241248782': 'ioniq 5',
    '01241228177': 'porter EV'
}

# Get folder path based on platform
if platform.system() == "Windows":
    folder_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\trip_by_trip')
elif platform.system() == "Darwin":
    folder_path = os.path.normpath(
        '/Users/wsong/Documents/KENTECH/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip')
else:
    print("Unknown system.")

# Get list of files
file_lists = get_file_list(folder_path)

# For each vehicle type, select a random file and plot
for key, vehicle in vehicle_types.items():
    # Get the list of files that match the current key
    vehicle_files = [file for file in file_lists if key in file]

    # Check if vehicle_files is empty
    if not vehicle_files:
        print(f"No files found for {vehicle}")
        continue

    # Choose a random file from the matched files
    selected_file = random.choice(vehicle_files)
    print(f"Selected File for {vehicle}: {selected_file}")

    # Read the selected file
    selected_df = pd.read_csv(os.path.join(folder_path, selected_file))
    selected_df['Residual'] = (selected_df['Power_IV'] - selected_df['Power'])
    mean_residual = selected_df['Residual'].mean()

    # Plotting for the selected file
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    sns.kdeplot(selected_df['Residual'], label=vehicle, fill=True)
    plt.axvline(mean_residual, color='r', linestyle='--')
    plt.text(mean_residual + 0.1, plt.ylim()[1] * 0.9, f"Mean: {mean_residual:.2f}", color='r')

    # Add the file name outside the plot, at the bottom right
    plt.annotate(selected_file,
                 xy=(1, -0.1),
                 xycoords='axes fraction',
                 fontsize=8,
                 ha='right',
                 va='center',
                 bbox=dict(boxstyle='square,pad=0', fc='white', ec='none'))

    plt.xlabel('Residual(kW)')
    plt.ylabel('Density')
    plt.title(f'Density Plot of Residuals (Data-Model) for {vehicle}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot energy comparison for the selected file
    t = pd.to_datetime(selected_df['time'], format='%Y-%m-%d %H:%M:%S')
    t_diff = t.diff().dt.total_seconds().fillna(0)
    t_diff = np.array(t_diff.fillna(0))
    t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

    bms_power = selected_df['Power_IV']
    bms_power = np.array(bms_power)
    data_energy = bms_power * t_diff / 3600 / 1000
    data_energy_cumulative = data_energy.cumsum()

    model_power = selected_df['Power']
    model_power = np.array(model_power)
    model_energy = model_power * t_diff / 3600 / 1000
    model_energy_cumulative = model_energy.cumsum()

    energy_difference = data_energy_cumulative[-1] - model_energy_cumulative[-1]
    energy_difference = "{:.4f}".format(energy_difference)
    energy_difference = str(energy_difference)

    # Plot the comparison graph
    plt.figure(figsize=(10, 6))
    plt.xlabel('Time (minutes)')
    plt.ylabel('BMS Energy and Model Energy (kWh)')
    plt.plot(t_min, model_energy_cumulative, label='Model Energy (kWh)', color='tab:blue')
    plt.plot(t_min, data_energy_cumulative, label='BMS Energy (kWh)', color='tab:red')

    # Add date, file name, and total time
    date = t.iloc[0].strftime('%Y-%m-%d')
    total_time = (t.iloc[-1] - t.iloc[0]).total_seconds()
    total_time_str = str(total_time)
    plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right', color='black')
    plt.text(0, 1, 'File: ' + selected_file, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='left', color='black')
    plt.text(0, 0, 'Total Time: ' + total_time_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='left', color='black')
    plt.text(1, 0, 'Energy Difference:' + energy_difference, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', color='black')

    plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
    plt.title('Model Energy vs. BMS Energy')
    plt.tight_layout()
    plt.show()
