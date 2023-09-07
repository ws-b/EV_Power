import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
def linear_func(v, acc, a, b):
    return 1 + a * v + b * acc

def objective(params, speed, acc, Power, Power_IV):
    a, b = params
    fitting_power = Power * linear_func(speed, acc, a, b)
    costs = ((fitting_power - Power_IV) ** 2).sum()
    return costs

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

        # 병합된 데이터프레임으로 fitting을 진행하여 파라미터를 추정합니다.
        speed = combined_df['speed']
        temp = combined_df['ext_temp']
        acc = combined_df['acceleration']
        Power = combined_df['Power']
        Power_IV = combined_df['Power_IV']

        initial_guess = [0,0]
        result = minimize(objective, initial_guess, args=(speed, acc, Power, Power_IV), method='BFGS')
        a, b = result.x

        # 해당 키의 모든 CSV 파일에 대해서 fitting을 진행합니다.
        for file in tqdm(files):
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            data['Power_fit'] = data['Power'] * linear_func(data['speed'], data['acceleration'], a, b)
            data.to_csv(file_path, index=False)

    print("Fitting 완료")

def fitting_multistart(file_lists, folder_path, num_starts=10):
    grouped_files = defaultdict(list)
    for file in file_lists:
        key = file[:11]
        grouped_files[key].append(file)

    costs = defaultdict(float)  # Store the costs for each (a, b)

    for key, files in grouped_files.items():
        selected_files = np.random.choice(files, size=len(files) // 2, replace=False)
        list_of_dfs = [pd.read_csv(os.path.join(folder_path, f)) for f in selected_files]
        combined_df = pd.concat(list_of_dfs, ignore_index=True).sort_values(by='time', ignore_index=True)

        speed = combined_df['speed']
        acc = combined_df['acceleration']
        Power = combined_df['Power']
        Power_IV = combined_df['Power_IV']

        # First optimization with initial guess [0,0]
        initial_guess = [0, 0]
        result = minimize(objective, initial_guess, args=(speed, acc, Power, Power_IV), method='BFGS')
        best_a, best_b = result.x
        best_cost = result.fun

        # Generate multiple initial guesses based on range
        a_range = np.linspace(-abs(10 * best_a), abs(10 * best_a), num_starts)
        b_range = np.linspace(-abs(10 * best_b), abs(10 * best_b), num_starts)

        for a_guess in a_range:
            for b_guess in b_range:
                result = minimize(objective, [a_guess, b_guess], args=(speed, acc, Power, Power_IV), method='BFGS')
                costs[(a_guess, b_guess)] = result.fun  # Store the cost for this (a, b) combination
                if result.fun < best_cost:
                    best_a, best_b = result.x
                    best_cost = result.fun

        for file in tqdm(files):
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            data['Power_fit'] = data['Power'] * linear_func(data['speed'], data['acceleration'], best_a, best_b)
            data.to_csv(file_path, index=False)

    print("Fitting 완료")
    return best_a, best_b, costs  # Return the costs as well

def plot_contour(a_range, b_range, costs):
    A, B = np.meshgrid(a_range, b_range)
    Z = np.array([[costs.get((a, b), np.nan) for a in a_range] for b in b_range])

    plt.contourf(A, B, Z, 50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('a')
    plt.ylabel('b')
    plt.title('Objective Function Landscape')
    plt.show()


def visualize_all_files(file_list, folder_path):
    all_dfs = []

    for file in tqdm(file_list):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        all_dfs.append(data)

    combined_data = pd.concat(all_dfs, ignore_index=True)

    plt.figure(figsize=(14, 7))
    plt.scatter(combined_data['Power_IV'], combined_data['Power_fit'], alpha=0.5)
    plt.plot([combined_data['Power_IV'].min(), combined_data['Power_IV'].max()],
             [combined_data['Power_IV'].min(), combined_data['Power_IV'].max()],
             color='red', linestyle='--', label='y=x line')
    plt.title("Comparison between Power_IV and Fitted Power")
    plt.xlabel('Power_IV')
    plt.ylabel('Power_fit')
    plt.legend()
    plt.show()