import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def plot_energy(file_lists, folder_path, Target):
    print("Plotting Energy, Put Target : model, data, comparison, stacked, altitude, d_altitude")
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

        if Target == 'model':
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

        elif Target == 'data':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('BMS Energy (kWh)')
            plt.plot(t_min, data_energy_cumulative, label='BMS Energy (kWh)', color='tab:blue')

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('BMS Energy')
            plt.tight_layout()
            plt.show()

        elif Target == 'comparison':
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
            plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Model Energy vs. BMS Energy')
            plt.tight_layout()
            plt.show()

        elif Target == 'stacked':
            A = data['A'] / 1000
            B = data['B'] / 1000
            C = data['C'] / 1000
            D = data['D'] / 1000
            E = data['E'] / 1000

            plt.figure(figsize=(12, 6))

            plt.stackplot(t_min, A, B, C, D, E,
                          labels=['A (First)', 'B (Second)', 'C (Third)', 'D (Accel)', 'E (Aux,Idle)'],
                          colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'], edgecolor=None)
            plt.title('Power Graph Term by Term')
            plt.xlabel('Time')
            plt.ylabel('Power (W)')
            plt.legend(loc='upper left')

            plt.show()

        elif Target == 'altitude' and 'altitude' in data.columns:
            # 고도 데이터
            altitude = np.array(data['altitude'])

            # 그래프 그리기
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # 첫 번째 y축 (왼쪽): 에너지 데이터
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Energy (kWh)')
            ax1.plot(t_min, model_energy_cumulative, label='Model Energy (kWh)', color='tab:red')
            ax1.plot(t_min, data_energy_cumulative, label='BMS Energy (kWh)', color='tab:blue')
            ax1.tick_params(axis='y')

            # 두 번째 y축 (오른쪽): 고도 데이터
            ax2 = ax1.twinx()
            ax2.set_ylabel('Altitude (m)', color='tab:green')  # 오른쪽 y축 레이블
            # ax2.set_ylim([0, 2000])
            ax2.plot(t_min, altitude, label='Altitude (m)', color='tab:green')
            ax2.tick_params(axis='y', labelcolor='tab:green')

            # 파일과 날짜 추가
            date = t.iloc[0].strftime('%Y-%m-%d')
            fig.text(0.99, 0.01, date, horizontalalignment='right', color='black', fontsize=12)
            fig.text(0.01, 0.99, 'File: ' + file, verticalalignment='top', color='black', fontsize=12)

            # 범례와 타이틀
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            plt.title('Model Energy vs. BMS Energy and Altitude')

            # 그래프 출력
            plt.tight_layout()
            plt.show()

        else:
            print("Invalid Target")

def plot_power(file_lists, folder_path, Target):
    print("Plotting Power, Put Target : model, data, comparison, difference, d_altitude")
    for file in tqdm(file_lists[31:35]):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # Convert time difference to minutes

        bms_power = data['Power_IV'] / 1000
        model_power = data['Power'] / 1000
        power_diff = (data['Power_IV'] - data['Power']) / 1000
        # A_power = data['A'] / 1000
        # B_power = data['B'] / 1000
        # C_power = data['C'] / 1000
        # D_power = data['D'] / 1000
        # E_power = data['E'] / 1000

        if Target == 'model':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('BMS Power and Model Power (kW)')
            plt.plot(t_min, bms_power, label='BMS Power (kW)', color='tab:blue')


            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Model Power vs. BMS Power')
            plt.tight_layout()
            plt.show()

        elif Target == 'data':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('BMS Power and Model Power (kW)')
            plt.plot(t_min, bms_power, label='BMS Power (kW)', color='tab:blue')

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Model Power vs. BMS Power')
            plt.tight_layout()
            plt.show()

        elif Target == 'comparison':
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
            plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Model Power vs. BMS Power')
            plt.tight_layout()
            plt.show()

        elif Target == 'difference':
            # Plot the comparison graph
            plt.figure(figsize=(10, 6))  # Set the size of the graph
            plt.xlabel('Time (minutes)')
            plt.ylabel('BMS Power - Model Power (kW)')
            plt.plot(t_min, power_diff, label='BMS Power - Model Power (kW)', color='tab:blue')

            # Add date and file name
            date = t.iloc[0].strftime('%Y-%m-%d')
            plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('BMS Power & Model Power Difference')
            plt.tight_layout()
            plt.show()

        elif Target == 'd_altitude' and 'delta altitude' in data.columns:
            # 고도 데이터
            d_altitude = np.array(data['delta altitude'])

            # 그래프 그리기
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # 첫 번째 y축 (왼쪽): 에너지 데이터
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Energy (kWh)')
            ax1.plot(t_min, bms_power, label='BMS Power (kW)', color='tab:blue')
            ax1.plot(t_min, model_power, label='model Power (kW)', color='tab:red')
            ax1.tick_params(axis='y')

            # 두 번째 y축 (오른쪽): 고도 데이터
            ax2 = ax1.twinx()
            ax2.set_ylabel('Altitude (m)', color='tab:green')  # 오른쪽 y축 레이블
            ax2.plot(t_min, d_altitude, label='Altitude (m)', color='tab:green')
            ax2.tick_params(axis='y', labelcolor='tab:green')

            # 파일과 날짜 추가
            date = t.iloc[0].strftime('%Y-%m-%d')
            fig.text(0.99, 0.01, date, horizontalalignment='right', color='black', fontsize=12)
            fig.text(0.01, 0.99, 'File: ' + file, verticalalignment='top', color='black', fontsize=12)

            # 범례와 타이틀
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            plt.title('Model Energy vs. BMS Energy and Delta Altitude')

            # 그래프 출력
            plt.tight_layout()
            plt.show()