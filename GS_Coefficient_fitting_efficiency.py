import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import zscore
import matplotlib.pyplot as plt
from tqdm import tqdm

win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\Ioniq5'

folder_path = os.path.normpath(win_folder_path)
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# Split file lists into training and testing
np.random.shuffle(file_lists)  # Randomize file order
training_files = file_lists[:len(file_lists) // 2]
testing_files = file_lists[len(file_lists) // 2:]
class Vehicle:
    def __init__(self, mass, load, Ca, Cb, Cc, aux, idle, a, b, E):
        self.mass = mass
        self.load = load
        self.Ca = Ca * 4.44822
        self.Cb = Cb * 4.44822 * 2.237
        self.Cc = Cc * 4.44822 * (2.237 ** 2)
        self.eff = a * E + b * E**2
        self.aux = aux
        self.idle = idle

def model_energy(params, data, E):  # Add 'E' as an argument
    a, b = params
    v = data['emobility_spd_m_per_s'].tolist()
    a = data['acceleration'].tolist()
    EV = Vehicle(2268, 0, 34.342, 0.21928, 0.022718, 870, 100, a, b, E)
    inertia = 0.05
    g = 9.18
    F = []
    E = []
    for velocity in v:
        F.append(EV.Ca + EV.Cb * velocity + EV.Cc * velocity * velocity)
    for i in range(len(a)):
        if a[i] >= -0.000001:
            F[i] += ((1 + inertia) * (EV.mass + EV.load) * a[i])
            E.append(F[i] * v[i])
        else:
            F[i] += ((((1 + inertia) * (EV.mass + EV.load) * a[i])) + ((1 + inertia) * (EV.mass + EV.load) * abs(a[i]) / np.exp(0.04111 / abs(a[i]))))
            E.append(F[i] * v[i])
        if v[i] <= 0.5:
            E[i] += (EV.aux + EV.idle)
        else:
            E[i] += EV.aux
    E = [i / 1800000 for i in E]
    return E * EV.eff


def objective(params, data, actual_energy):
    predicted_energy = model_energy(params, data, E)  # Pass 'E' here
    # Take cumulative sum of predicted energy
    predicted_energy_cumulative = np.cumsum(predicted_energy)
    # Compute the squared error of the cumulative energies
    return np.sum((predicted_energy_cumulative - actual_energy) ** 2)

optimized_params = []
for file in tqdm(training_files):
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    CHARGE = data['trip_chrg_pw'].tolist()
    DISCHARGE = data['trip_dischrg_pw'].tolist()
    actual_energy = np.array(DISCHARGE) - np.array(CHARGE)
    initial_guess = [1, 1]
    bounds = [(-1, 1), (-1, 1)]
    result = minimize(objective, initial_guess, args=(data, actual_energy), bounds = bounds)
    optimized_params.append(result.x)

# Compute average parameters
optimized_params_array = np.array(optimized_params)
z_scores = zscore(optimized_params_array)
optimized_params_array = optimized_params_array[(np.abs(z_scores) < 2).all(axis=1)]  # Remove outliers
average_params = np.mean(optimized_params_array, axis=0)

# Create a DataFrame from the optimized parameters
df_params = pd.DataFrame(optimized_params_array, columns=['a', 'b'])

# Predict energy on testing data
for file in tqdm(testing_files[:10]):  # Select first 10 files
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

    # Convert time to minutes
    t = (t - t.min()).dt.total_seconds() / 60

    CHARGE = data['trip_chrg_pw'].tolist()
    DISCHARGE = data['trip_dischrg_pw'].tolist()

    # Convert energy to kWh
    actual_energy = (np.array(DISCHARGE) - np.array(CHARGE))
    predicted_energy = np.array(model_energy(average_params, data))
    predicted_energy_cumulative = np.cumsum(predicted_energy)  # Taking cumulative sum of predicted energy

    plt.figure(figsize=(10,6))
    plt.plot(t, actual_energy, label='Actual Energy')  # Plotting actual energy directly
    plt.plot(t, predicted_energy_cumulative, label='Predicted Energy')  # Plotting cumulative predicted energy

    # Add filename to the left of the graph
    plt.text(0, 1, 'File: ' + file, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             horizontalalignment='left')

    plt.xlabel('Time (minutes)')
    plt.ylabel('Cumulative Energy (kWh)')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))  # Moving legend slightly down
    plt.show()

# Plotting the optimized parameters
fig, axs = plt.subplots(3, 1, figsize=(10,15))

# Plot a
axs[0].plot(df_params.index, df_params['a'], label='a', marker='o', color='b')
axs[0].set_ylabel('a')
axs[0].legend()

# Plot b
axs[1].plot(df_params.index, df_params['b'], label='b', marker='o', color='r')
axs[1].set_ylabel('b')

axs[1].set_xlabel('Iteration')
axs[1].legend()

plt.suptitle('Estimated Parameters over Iterations')
plt.show()
