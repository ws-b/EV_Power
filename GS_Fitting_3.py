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

def linear_regression(X, y):
    X = np.column_stack([np.ones(X.shape[0]), X])
    beta = inv(X.T @ X) @ X.T @ y
    return beta

def bootstrap_p_value(X, y, num_iterations=10000):
    beta_original = linear_regression(X, y)
    coeffs_bootstrap = []

    for _ in range(num_iterations):
        X_resampled, y_resampled = resample(X, y)
        beta_bootstrap = linear_regression(X_resampled, y_resampled)
        coeffs_bootstrap.append(beta_bootstrap[1:])

    coeffs_bootstrap = np.array(coeffs_bootstrap)
    p_values = []
    for i in range(coeffs_bootstrap.shape[1]):
        coeff_dist = coeffs_bootstrap[:, i]
        coeff_orig = beta_original[i+1]
        p_value = (np.abs(coeff_dist) >= np.abs(coeff_orig)).mean()
        p_values.append(p_value)

    return p_values

def fitting_with_p_values(file_lists, folder_path):
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
        result = basinhopping(objective, initial_guess,
                              minimizer_kwargs={"args": (speed, acc, Power, Power_IV), "method": "BFGS"})
        a, b = result.x

        X = np.column_stack([speed, acc])
        y = Power_IV - Power * linear_func(speed, acc, a, b)
        p_values_bootstrap = bootstrap_p_value(X, y)

        for file in tqdm(files):
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            data['speed_nmz'] = normalize(data['speed'])
            data['acceleration_nmz'] = normalize(data['acceleration'])
            data['Power_fit'] = data['Power'] * linear_func(data['speed_nmz'], data['acceleration_nmz'], a, b)
            data.to_csv(file_path, index=False)

        visualize_objective(combined_df, objective, a, b, p_values_bootstrap)

        print(f"P-values (bootstrap) for {key} - a: {p_values_bootstrap[0]:.5f}, b: {p_values_bootstrap[1]:.5f}")

    print("Fitting 완료")

def visualize_objective(data, objective_func, a, b, p_values):
    speed = data['speed']
    acc = data['acceleration']
    Power = data['Power']
    Power_IV = data['Power_IV']

    a_values = np.linspace(-10, 10, 200)
    b_values = np.linspace(-10, 10, 200)
    A, B = np.meshgrid(a_values, b_values)

    Z = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Z[i, j] = objective_func([A[i, j], B[i, j]], speed, acc, Power, Power_IV)

    Z = (Z - Z.min()) / (Z.max() - Z.min())

    fig, ax = plt.subplots(figsize=(10, 7))
    cp = ax.contourf(A, B, Z, cmap='viridis', levels=50)
    plt.colorbar(cp, ax=ax, label='Normalized Objective Function Value')
    ax.scatter(a, b, color='red', marker='o', s=10, label=f'Optimal Parameters (a, b)\n(a={a:.3f}, b={b:.3f})')
    ax.set_xlabel('Parameter a')
    ax.set_ylabel('Parameter b')
    ax.set_title('Normalized Objective Function Landscape')
    ax.legend()

    # Add the p-values outside the graph on the right top corner
    ax.annotate(f'p-value for a: {p_values[0]:.5f}\np-value for b: {p_values[1]:.5f}',
                xy=(1.05, 1.07), xycoords='axes fraction', fontsize=10, ha='left', va='top')
    plt.show()