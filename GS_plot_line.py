import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
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

        plt.stackplot(t_min, A, B, C, D, E, labels=['A (First)', 'B (Second)', 'C (Third)', 'D (Accel)', 'E (Aux,Idle)'], colors=['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0'], edgecolor=None)
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
    for file in tqdm(file_lists[31:35]):
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

        total_drive_time = t_min.iloc[-1] - t_min.iloc[0]
        regen_brake_time = t_min_masked.iloc[-1] - t_min_masked.iloc[0]
        regen_brake_ratio = regen_brake_time / total_drive_time

        print(f"Regenerative braking time 비율은{regen_brake_ratio}")

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        ax1 = plt.gca()
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Power (kW)')

        # Use the masked data for the plot
        ax1.plot(t_min_masked, data_energy_masked, label='BMS Power (kW)', color='tab:red')
        ax1.plot(t_min_masked, D_masked, label='D term Power (kW)', color='tab:blue')

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

def plot_power_comparison(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        bms_power = data['Power_IV'] / 1000
        model_power = data['Power'] / 1000
        # A_power = data['A'] / 1000
        # B_power = data['B'] / 1000
        # C_power = data['C'] / 1000
        # D_power = data['D'] / 1000
        # E_power = data['E'] / 1000

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        plt.xlabel('Time (minutes)')
        plt.ylabel('BMS Power and Model Power (kW)')
        plt.plot(t_min, bms_power, label='BMS Power (kW)', color='tab:blue')
        plt.plot(t_min, model_power, label='model Power (kW)', color='tab:red')
        # plt.plot(t_min, A_power, label='v Term (kW)', color='tab:orange')
        # plt.plot(t_min, B_power, label='v^2 Term (kW)', color='tab:purple')
        # plt.plot(t_min, C_power, label='v^3 Term (kW)', color='tab:pink')
        # plt.plot(t_min, D_power, label='Acceleration Term (kW)', color='tab:green')
        # plt.plot(t_min, E_power, label='Aux/Idle Term (kW)', color='tab:brown')

        # Add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
        plt.title('Model Power vs. BMS Power')
        plt.tight_layout()
        plt.show()
def plot_power_comparison_enlarge(file_lists, folder_path, threshold=0.5):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        bms_power = data['Power_IV'] / 1000
        bms_power = np.array(bms_power)
        data_energy = bms_power * t_diff / 3600
        data_energy_cumulative = data_energy.cumsum()

        model_power = data['Power'] / 1000
        model_power = np.array(model_power)
        model_energy = model_power * t_diff / 3600
        model_energy_cumulative = model_energy.cumsum()

        # A_power = data['A'] / 1000
        # B_power = data['B'] / 1000
        # C_power = data['C'] / 1000
        # D_power = data['D'] / 1000
        # E_power = data['E'] / 1000

        # Calculate the absolute difference between BMS and model energy cumulative
        power_difference = np.abs(data_energy_cumulative - model_energy_cumulative)

        # Find the first index where the difference exceeds the threshold
        start_index = np.argmax(power_difference > threshold)
        if start_index > 0:
            start_time = t_min[start_index]
            interval = 5  # 5 minutes interval
            end_time = start_time + interval

            while start_time < t_min.iloc[-1]:
                # Plot the comparison graph
                plt.figure(figsize=(10, 6))  # Set the size of the graph
                plt.xlabel('Time (minutes)')
                plt.ylabel('BMS Power and Model Power (kW)')
                plt.plot(t_min, bms_power, label='BMS Power (kW)', color='tab:blue')
                # plt.plot(t_min, model_power, label='model Power (kW)', color='tab:red')

                # plt.plot(t_min, A_power, label='v Term (kW)', color='tab:orange')
                # plt.plot(t_min, B_power, label='v^2 Term (kW)', color='tab:purple')
                # plt.plot(t_min, C_power, label='v^3 Term (kW)', color='tab:pink')
                # plt.plot(t_min, D_power, label='Acceleration Term (kW)', color='tab:green')
                # plt.plot(t_min, E_power, label='Aux/Idle Term (kW)', color='tab:brown')

                # Add date and file name
                date = t.iloc[0].strftime('%Y-%m-%d')
                plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                         verticalalignment='top', horizontalalignment='right', color='black')
                plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                         verticalalignment='top', horizontalalignment='left', color='black')

                # Set the x-axis limit to zoom in on the divergence for 5 minutes
                plt.xlim(start_time, end_time)
                plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
                plt.title('Model Power vs. BMS Power')
                plt.tight_layout()
                plt.show()

                # Move to the next 5-minute window
                start_time += interval
                end_time += interval


def plot_power_diff(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        energy_diff = (data['Power_IV'] - data['Power']) / 1000

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        plt.xlabel('Time (minutes)')
        plt.ylabel('BMS Power - Model Power (kW)')
        plt.plot(t_min, energy_diff, label='BMS Power - Model Power (kW)', color='tab:blue')

        # Add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
        plt.title('BMS Power & Model Power Difference')
        plt.tight_layout()
        plt.show()

def plot_correlation(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        data['energy_diff'] = (data['Power_IV'] - data['Power']) / 1000

        # Compute the correlation between energy_diff and each of A_power, B_power, C_power, D_power, E_power
        correlations = data[['energy_diff', 'A', 'B', 'C', 'D', 'E']].corr()

        # Create a heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of energy_diff and Power Terms')
        plt.show()
