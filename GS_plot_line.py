import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_stacked_graph(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        A = data['A'] / 1000
        B = data['B'] / 1000
        C = data['C'] / 1000
        D = data['D'] / 1000
        E = data['E'] / 1000

        plt.figure(figsize=(12, 6))

        plt.stackplot(t_min, A, B, C, D, E, labels=['A(First)', 'B(Second)', 'C(Third)', 'D(Accel)', 'E(AUX)'], colors=['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0'], edgecolor=None)
        plt.title('Power Graph Term by Term')
        plt.xlabel('Time')
        plt.ylabel('Power (W)')
        plt.legend(loc='upper left')

        plt.show()

def plot_model_energy(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        model_power = data['Power']
        model_power = np.array(model_power)
        model_energy = model_power * t_diff / 3600 / 1000
        model_energy_cumulative = model_energy.cumsum()

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        plt.xlabel('Time (minutes)')
        plt.ylabel('Model Energy (kWh)')
        plt.plot(t_min, model_energy_cumulative, label='Model Energy (kWh)', color='tab:blue')

        # Add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
        plt.title('Model Energy over time')
        plt.tight_layout()
        plt.show()
def plot_bms_energy(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        plt.xlabel('Time (minutes)')
        plt.ylabel('BMS Energy (kWh)')
        plt.plot(t_min, data_energy_cumulative, label='BMS Energy (kWh)', color='tab:red')

        # Add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
        plt.title('BMS Energy')
        plt.tight_layout()
        plt.show()
def plot_energy_comparison(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600 / 1000
        data_energy_cumulative = data_energy.cumsum()

        model_power = data['Power']
        model_power = np.array(model_power)
        model_energy = model_power * t_diff / 3600 / 1000
        model_energy_cumulative = model_energy.cumsum()

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        plt.xlabel('Time (minutes)')
        plt.ylabel('BMS Energy and Model Energy (kWh)')
        plt.plot(t_min, model_energy_cumulative, label='Model Energy (kWh)', color='tab:blue')
        plt.plot(t_min, data_energy_cumulative, label='BMS Energy (kWh)', color='tab:red')

        # Add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
        plt.title('Model Energy vs. BMS Energy')
        plt.tight_layout()
        plt.show()

def plot_regen_brake_effect(file_lists, folder_path):
    for file in tqdm(file_lists[30:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        bms_power = data['Power_IV']
        bms_power = np.array(bms_power)
        data_energy = bms_power / 1000

        # Extract D values and convert from W to kW
        D = data['D'] / 1000  # Convert from W to kW

        # Extract acceleration values
        acceleration = data['acceleration']

        # Create a boolean mask for rows where Power_IV is negative
        mask = bms_power < 0

        # Apply the mask to t_min, D, data_energy_cumulative, and acceleration
        t_min_masked = t_min[mask]
        D_masked = D[mask]
        data_energy_masked = data_energy[mask]
        acceleration_masked = acceleration[mask]

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        ax1 = plt.gca()
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Power (kW)')

        # Use the masked data for the plot
        ax1.plot(t_min_masked, data_energy_masked, label='BMS Power (kW)', color='tab:red')
        ax1.plot(t_min_masked, D_masked.cumsum(), label='D Power (kW)', color='tab:blue')

        ax2 = ax1.twinx()  # Create a second y-axis
        ax2.set_ylabel('Acceleration')  # Add label to the second y-axis
        ax2.plot(t_min_masked, acceleration_masked, label='Acceleration', color='tab:green', linestyle='--')

        # Add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        ax1.text(1, 1, date, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        ax1.text(0, 1, 'File: '+file, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0, 0.98))

        plt.title('BMS Data and D Power with Acceleration')
        plt.tight_layout()
        plt.show()