import os
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
def split_data(file_lists, train_ratio=0.5, test_size=4):
    # Randomly split files into training and validation sets
    train_files, validation_files = train_test_split(file_lists, test_size=1 - train_ratio, random_state=42)

    # Randomly select specific files for testing within the validation set
    test_files = np.random.choice(validation_files, test_size, replace=False)

    return train_files, test_files
def linear_func(v, a, b):
    return a + b * v

def objective(params, speed, Power, Power_IV):
    a, b = params
    fitting_power = Power * linear_func(speed, a, b)
    return ((fitting_power - Power_IV) ** 2).sum()

def fit_power(data):
    speed = data['speed']
    Power = data['Power']
    Power_IV = data['Power_IV']

    # 초기 추정값
    initial_guess = [0, 0]

    # 최적화 수행
    result = minimize(objective, initial_guess, args=(speed, Power, Power_IV))

    a, b = result.x

    # 최적화된 Power 값을 별도의 컬럼으로 저장
    data['Power_fit'] = Power * linear_func(speed, a, b)

    return a, b, data

def plot_fit_energy_comparison(file_lists, folder_path):
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

        model_power = data['Power_fit']
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

def plot_fit_power_comparison(file_lists, folder_path):
    for file in tqdm(file_lists):
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
def fit_parameters(train_files, folder_path):
    a_values = []
    b_values = []
    for file in tqdm(train_files):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        a, b, _ = fit_power(data)
        a_values.append(a)
        b_values.append(b)

    a_avg = sum(a_values) / len(a_values)
    b_avg = sum(b_values) / len(b_values)
    return a_avg, b_avg

def apply_fitting(test_files, folder_path, a_avg, b_avg):
    for file in tqdm(test_files):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        data['Power_fit'] = data['Power'] * linear_func(data['speed'], a_avg, b_avg)
        data.to_csv(os.path.join(folder_path, file), index=False)

def fitting_and_plotting(file_lists, folder_path):
    # 훈련 및 테스트 데이터 분리
    train_files, test_files = split_data(file_lists)

    # 훈련 데이터에서 a, b 파라미터 최적화
    a_avg, b_avg = fit_parameters(train_files, folder_path)

    # 테스트 데이터에 대한 Power_fit 계산 및 저장
    apply_fitting(test_files, folder_path, a_avg, b_avg)

    # 테스트 데이터에 대한 그래프 그리기
    plot_fit_energy_comparison(test_files, folder_path)
    plot_fit_power_comparison(test_files, folder_path)
    plot_model_energy_dis(test_files, folder_path)
    print("Done")

def plot_model_energy_dis(file_lists, folder_path):
    all_distance_per_total_energy = []

    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff.fillna(0))

        v = data['speed']
        v = np.array(v)

        model_power = data['Power_fit']
        model_power = np.array(model_power)
        model_energy = model_power * t_diff / 3600 / 1000
        model_energy_cumulative = model_energy.cumsum()

        distance = v * t_diff
        total_distance = distance.cumsum()

        # calculate Total distance / Total Energy for each file (if Total Energy is 0, set the value to 0)
        distance_per_total_energy = (total_distance[-1] / 1000) / (model_energy_cumulative[-1]) if model_energy_cumulative[-1] != 0 else 0

        # collect all distance_per_total_Energy values for all files
        all_distance_per_total_energy.append(distance_per_total_energy)

    # plot histogram for all files
    hist_data = sns.histplot(all_distance_per_total_energy, bins='auto', color='gray', kde=False)

    # plot vertical line for mean value
    mean_value = np.mean(all_distance_per_total_energy)
    plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')

    # display mean value
    plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red', fontsize=12)

    # display total number of samples
    total_samples = len(all_distance_per_total_energy)
    plt.text(0.95, 0.95, f'Total Samples: {total_samples}', horizontalalignment='right',
             verticalalignment='top', transform=plt.gca().transAxes)

    # set x-axis range (from 0 to 25)
    plt.xlim(0, 25)
    plt.xlabel('Total Distance / Total Model Energy (km/kWh)')
    plt.ylabel('Number of trips')
    plt.title('Total Distance / Total Model Energy Distribution')
    plt.grid(False)
    plt.show()
