import os
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping
from scipy.stats import t
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.utils import resample

def linear_func(v, acc, a, b):
    return 1 + a * v + b * acc

def objective(params, *args):
    a, b = params
    speed, acc, Power, Power_IV = args
    fitting_power = Power * linear_func(speed, acc, a, b)
    costs = ((fitting_power - Power_IV) ** 2).sum()
    return costs

def normalize(data):
    return data / abs(data).mean()

def compute_manual_p_values(X, y):
    X = np.column_stack([np.ones(X.shape[0]), X])
    beta = inv(X.T @ X) @ X.T @ y
    residuals = y - X @ beta
    residual_sum_of_squares = residuals.T @ residuals
    var_beta = np.diagonal(residual_sum_of_squares / (X.shape[0] - X.shape[1]) * inv(X.T @ X))
    std_err_beta = np.sqrt(var_beta)
    t_stats = beta / std_err_beta
    df = X.shape[0] - X.shape[1]
    p_values = 2 * (1 - t.cdf(np.abs(t_stats), df))
    return p_values[1:]

def fitting_with_manual_p_values(file_lists, folder_path):
    grouped_files = defaultdict(list)
    for file in file_lists:
        key = file[:11]
        grouped_files[key].append(file)

    for key, files in grouped_files.items():
        selected_files = np.random.choice(files, size=len(files) // 2, replace=False)
        list_of_dfs = [pd.read_csv(os.path.join(folder_path, f)) for f in selected_files]
        combined_df = pd.concat(list_of_dfs, ignore_index=True)
        combined_df = combined_df.sort_values(by='time', ignore_index=True)
        combined_df['speed_nmz'] = normalize(combined_df['speed'])
        combined_df['acceleration_nmz'] = normalize(combined_df['acceleration'])
        speed = combined_df['speed_nmz']
        acc = combined_df['acceleration_nmz']
        Power = combined_df['Power']
        Power_IV = combined_df['Power_IV']
        initial_guess = [0, 0]
        minimizer = {"method": "BFGS"}
        result = basinhopping(objective, initial_guess, minimizer_kwargs={"args": (speed, acc, Power, Power_IV), "method": "BFGS"})
        a, b = result.x
        X = np.column_stack([speed, acc])
        y = Power_IV - Power * linear_func(speed, acc, a, b)
        p_values_manual = compute_manual_p_values(X, y)
        for file in files:
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            data['speed_nmz'] = normalize(data['speed'])
            data['acceleration_nmz'] = normalize(data['acceleration'])
            data['Power_fit'] = data['Power'] * linear_func(data['speed_nmz'], data['acceleration_nmz'], a, b)
            data.to_csv(file_path, index=False)
        print(f"P-values (manual) for {key} - a: {p_values_manual[0]:.5f}, b: {p_values_manual[1]:.5f}")

    print("Fitting 완료")