import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
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
        plt.plot(t_min, model_energy_cumulative, label='Model Energy (kWh)', color='tab:red')

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
        plt.plot(t_min, data_energy_cumulative, label='BMS Energy (kWh)', color='tab:blue')

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
        plt.plot(t_min, model_energy_cumulative, label='Model Energy (kWh)', color='tab:red')
        plt.plot(t_min, data_energy_cumulative, label='BMS Energy (kWh)', color='tab:blue')

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

def plot_speed_power(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        speed = data['speed']
        acceleration = data['acceleration']
        bms_power = data['Power_IV'] / 1000
        model_power = data['Power'] / 1000

        # Plot the comparison graph
        fig, ax1 = plt.subplots(figsize=(10, 6))  # Create a subplot and get the first axis
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Power (kW), Speed (m/s)')
        ax1.set_ylim([-100, 100])
        line1, = ax1.plot(t_min, bms_power, label='BMS Power (kW)', color='lightblue')
        #line2, = ax1.plot(t_min, model_power, label='model Power (kW)', color='lightsalmon')
        line3, = ax1.plot(t_min, speed, label='Speed (m/s)', color='lightgreen')


        # # Create the second y-axis and plot acceleration
        # ax2 = ax1.twinx()
        # line4, = ax2.plot(t_min, acceleration, label='Acceleration (m/s²)', color='lightcoral')
        # ax2.set_ylabel('Acceleration (m/s²)')  # Add label for the second y-axis
        # ax2.set_ylim([-3, 3])

        # Combine legends
        #lines = [line1, line2, line3, line4]
        lines = [line1, line3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 0.97))

        # Add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: ' + file, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.title('Speed vs. BMS Power')
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
        plt.ylim([-100, 100])
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

def plot_fit_power_comparison(file_lists, folder_path):
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        bms_power = data['Power_IV'] / 1000
        model_power = data['Power_fit'] / 1000

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        plt.xlabel('Time (minutes)')
        plt.ylabel('BMS Power and Fitted Model Power (kW)')
        plt.plot(t_min, bms_power, label='BMS Power (kW)', color='tab:blue')
        plt.plot(t_min, model_power, label='Fitted Model Power (kW)', color='tab:red')
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
        plt.title('Fitted Model Power vs. BMS Power')
        plt.tight_layout()
        plt.show()

def plot_fit_energy_comparison(file_lists, folder_path):
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

        model_power = data['Power_fit']
        model_power = np.array(model_power)
        model_energy = model_power * t_diff / 3600 / 1000
        model_energy_cumulative = model_energy.cumsum()

        # Plot the comparison graph
        plt.figure(figsize=(10, 6))  # Set the size of the graph
        plt.xlabel('Time (minutes)')
        plt.ylabel('BMS Energy and Fitted Model Energy (kWh)')
        plt.plot(t_min, model_energy_cumulative, label='Fit Model Energy (kWh)', color='tab:blue')
        plt.plot(t_min, data_energy_cumulative, label='BMS Energy (kWh)', color='tab:red')

        # Add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
        plt.title('Fitted Model Energy vs. BMS Energy')
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


def plot_bms_energy_pdf(file_lists, folder_path, EV):
    # PDF 저장 경로 설정
    pdf_path = os.path.join(folder_path, f'{EV}_bms_energy_graphs.pdf')

    with PdfPages(pdf_path) as pdf:
        for file in tqdm(file_lists):
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

            # Plot the comparison graph for energy
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

            # Energy graph
            ax1.set_ylabel('BMS Energy and Model Energy (kWh)')
            ax1.plot(t_min, model_energy_cumulative, label='Model Energy (kWh)', color='tab:red')
            ax1.plot(t_min, data_energy_cumulative, label='BMS Energy (kWh)', color='tab:blue')
            ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.95))  # 레전드 위치 조정
            ax1.set_title('Model Energy vs. BMS Energy')

            # Speed and acceleration graph
            ax2.plot(t_min, data['speed'], label='Speed (m/s)', color='tab:green')
            ax3_ax2 = ax2.twinx()  # instantiate a second y-axis sharing the same x-axis
            ax3_ax2.plot(t_min, data['acceleration'], label='Acceleration (m/s^2)', color='tab:orange')
            ax2.set_ylabel('Speed (m/s)', color='tab:green')
            ax3_ax2.set_ylabel('Acceleration (m/s^2)', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:green')
            ax3_ax2.tick_params(axis='y', labelcolor='tab:orange')

            # Power graph
            ax3.plot(t_min, model_power, label='Model Power', color='tab:red')
            ax3.plot(t_min, bms_power, label='BMS Power', color='tab:blue')
            ax3.set_ylabel('Power')
            ax3.set_xlabel('Time (minutes)')
            ax3.legend(loc='upper left')

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            ax1.text(1, 1, date, transform=ax1.transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            ax1.text(0, 1, 'File: ' + file, transform=ax1.transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.tight_layout()
            # 현재 그래프를 PDF에 저장
            pdf.savefig()
            plt.close()

    print(f"Graphs saved in {pdf_path}")