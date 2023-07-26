import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_energy_comparison(file_lists, folder_path):
    # Plot graphs for each file
    for file in tqdm(file_lists):
        # Create file path
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # Extract time, Power, CHARGE, DISCHARGE
        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes
        Energy_IV = data['Energy_IV'].tolist()

        # Convert Power data to kWh
        Power_kWh = data['Energy']  # Convert kW to kWh considering the 2-second time interval

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        plt.xlabel('Time (minutes)')
        plt.ylabel('BMS Energy and Model Energy (kWh)')
        plt.plot(t_diff, Power_kWh, label='Model Energy (kWh)', color='tab:blue')
        plt.plot(t_diff, Energy_IV, label='BMS Energy (kWh)', color='tab:red')

        # Add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))  # Moving legend slightly down
        plt.title('Model Energy vs. BMS Energy')
        plt.tight_layout()  # Otherwise the right y-label is slightly clipped
        plt.show()
