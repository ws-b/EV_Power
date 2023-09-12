from scipy.optimize import basinhopping
from collections import defaultdict
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def linear_func(v, acc, a, b):
    return 1 + a * v + b * acc


def objective(params, *args):
    a, b = params
    speed, acc, Power, Power_IV = args
    fitting_power = Power * linear_func(speed, acc, a, b)
    costs = ((fitting_power - Power_IV) ** 2).sum()
    return costs


def min_max_normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def fitting(file_lists, folder_path):
    # CSV 파일들을 그룹화
    grouped_files = defaultdict(list)
    for file in file_lists:
        key = file[:11]
        grouped_files[key].append(file)

    for key, files in grouped_files.items():
        # 파일 목록의 50%를 랜덤하게 선택합니다.
        selected_files = np.random.choice(files, size=len(files) // 2, replace=False)

        list_of_dfs = [pd.read_csv(os.path.join(folder_path, f)) for f in selected_files]

        # 선택된 파일들의 데이터프레임을 병합합니다.
        combined_df = pd.concat(list_of_dfs, ignore_index=True)
        combined_df = combined_df.sort_values(by='time', ignore_index=True)

        # Normalize speed and acceleration
        combined_df['speed_nmz'] = min_max_normalize(combined_df['speed'])
        combined_df['acceleration_nmz'] = min_max_normalize(combined_df['acceleration'])
        combined_df['ext_temp_nmz'] = min_max_normalize(combined_df['ext_temp'])

        # 병합된 데이터프레임으로 fitting을 진행하여 파라미터를 추정합니다.
        speed = combined_df['speed_nmz']
        acc = combined_df['acceleration_nmz']
        # acc = combined_df['ext_temp_nmz']
        Power = combined_df['Power']
        Power_IV = combined_df['Power_IV']

        initial_guess = [0, 0]
        minimizer = {"method": "BFGS"}
        result = basinhopping(objective, initial_guess,
                              minimizer_kwargs={"args": (speed, acc, Power, Power_IV), "method": "BFGS"})
        a, b = result.x

        # 해당 키의 모든 CSV 파일에 대해서 fitting을 진행합니다.
        for file in tqdm(files):
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            data['speed_nmz'] = min_max_normalize(data['speed'])
            data['acceleration_nmz'] = min_max_normalize(data['acceleration'])
            data['ext_temp_nmz'] = min_max_normalize(data['ext_temp'])
            data['Power_fit'] = data['Power'] * linear_func(data['speed_nmz'], data['acceleration_nmz'], a, b)
            data.to_csv(file_path, index=False)

        # Visualize for the current key
        visualize_objective(combined_df, objective, a, b)

    print("Fitting 완료")

def visualize_objective(data, objective_func, a, b):
    # Extract relevant columns from the data
    speed = data['speed']
    acc = data['acceleration']
    Power = data['Power']
    Power_IV = data['Power_IV']

    # Create a grid over the parameter space
    a_values = np.linspace(-10, 10, 200)  # Adjust bounds if needed
    b_values = np.linspace(-10, 10, 200)
    A, B = np.meshgrid(a_values, b_values)

    # Evaluate the objective function on the grid
    Z = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Z[i, j] = objective_func([A[i, j], B[i, j]], speed, acc, Power, Power_IV)

    # Normalize the objective function values
    Z = (Z - Z.min()) / (Z.max() - Z.min())

    # Plot
    plt.figure(figsize=(10, 7))
    cp = plt.contourf(A, B, Z, cmap='viridis', levels=50)
    plt.colorbar(cp, label='Normalized Objective Function Value')
    plt.scatter(a, b, color='red', marker='o', s=10, label=f'Optimal Parameters (a, b)\n(a={a:.3f}, b={b:.3f})')
    plt.xlabel('Parameter a')
    plt.ylabel('Parameter b')
    plt.title('Normalized Objective Function Landscape')
    plt.legend()
    plt.show()