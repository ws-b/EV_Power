import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
def linear_func(v, T, a, b, c):
    return a + b * v + c * T

def objective(params, speed, temp, Power, Power_IV):
    a, b, c = params
    fitting_power = Power * linear_func(speed, temp, a, b, c)
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

        # # 병합된 데이터프레임을 CSV 파일로 저장합니다.
        # combined_df_path = os.path.join(folder_path, f"{key}.csv")
        # combined_df.to_csv(combined_df_path, index=False)

        # 병합된 데이터프레임으로 fitting을 진행하여 파라미터를 추정합니다.
        speed = combined_df['speed']
        temp = combined_df['ext_temp']
        Power = combined_df['Power']
        Power_IV = combined_df['Power_IV']

        initial_guess = [0,0,0]
        result = minimize(objective, initial_guess, args=(speed, temp, Power, Power_IV), method='BFGS')
        a, b, c = result.x

        # 해당 키의 모든 CSV 파일에 대해서 fitting을 진행합니다.
        for file in tqdm(files):
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            data['Power_fit'] = data['Power'] * linear_func(data['speed'], data['ext_temp'], a, b, c)
            data.to_csv(file_path, index=False)

    print("Fitting 완료")


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