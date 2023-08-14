import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm


def linear_func(v, T, a, b, c):
    return a + b * v + c * T

def objective(params, speed, temp, Power, Power_IV):
    a, b, c = params
    fitting_power = Power * linear_func(speed, temp, a, b, c)
    costs = ((fitting_power - Power_IV) ** 2).sum()
    return costs

def fit_power(file, folder_path, num_starts=10):
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)
    speed = data['speed']
    temp = data['ext_temp']
    Power = data['Power']
    Power_IV = data['Power_IV']

    best_result = None
    best_value = float('inf')

    for _ in range(num_starts):
        initial_guess = np.random.rand(3) * 10  # 초기값을 임의로 설정
        result = minimize(objective, initial_guess, args=(speed, temp, Power, Power_IV), method='BFGS')

        if result.fun < best_value:
            best_value = result.fun
            best_result = result

    a, b, c = best_result.x

    data['Power_fit'] = Power * linear_func(speed, temp, a, b, c)
    data.to_csv(file_path, index=False)

    return a, b, c

def fitting(file_lists, folder_path):
    for file in tqdm(file_lists):
        fit_power(file, folder_path)
    print("Done")