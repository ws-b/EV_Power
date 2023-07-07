import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\ioniq5'

folder_path = os.path.normpath(win_folder_path)
# Get all csv files
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

class Vehicle:
    def __init__(self, mass, load, Ca, Cb, Cc, aux, idle, eff):
        self.mass = mass
        self.load = load
        self.Ca = Ca * 4.44822
        self.Cb = Cb * 4.44822 * 2.237
        self.Cc = Cc * 4.44822 * (2.237 ** 2)
        self.eff = eff
        self.aux = aux
        self.idle = idle

def model_energy(params, data):
    Ca, Cb, Cc = params
    v = data['emobility_spd_m_per_s'].tolist()
    a = data['acceleration'].tolist()
    EV = Vehicle(2268, 0, Ca, Cb, Cc, 870, 100, 0.9)
    inertia = 0.05
    g = 9.18
    F = []
    E = []
    for velocity in v:
        F.append(EV.Ca + EV.Cb * velocity + EV.Cc * velocity * velocity)
    for i in range(len(a)):
        if a[i] >= -0.000001:
            F[i] += ((1 + inertia) * (EV.mass + EV.load) * a[i])
            E.append(F[i] * v[i] / EV.eff)
        else:
            F[i] += ((((1 + inertia) * (EV.mass + EV.load) * a[i])) + ((1 + inertia) * (EV.mass + EV.load) * abs(a[i]) / np.exp(0.04111 / abs(a[i]))))
            E.append(F[i] * v[i] / EV.eff)
        if v[i] <= 0.5:
            E[i] += (EV.aux + EV.idle)
        else:
            E[i] += EV.aux
    E = [i / 1800000 for i in E]
    return E


def objective(params, data, actual_energy):
    predicted_energy = model_energy(params, data)
    # Take cumulative sum of predicted energy
    predicted_energy_cumulative = np.cumsum(predicted_energy)
    # Compute the squared error of the cumulative energies
    return np.sum((predicted_energy_cumulative - actual_energy) ** 2)

optimized_params = []

for file in file_lists[25:30]:
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    CHARGE = data['trip_chrg_pw'].tolist()
    DISCHARGE = data['trip_dischrg_pw'].tolist()
    actual_energy = np.array(DISCHARGE) - np.array(CHARGE)
    initial_guess = [34.342, 0.21928, 0.022718]
    result = minimize(objective, initial_guess, args=(data, actual_energy))
    optimized_params.append(result.x)

for idx, file in enumerate(file_lists[25:30]):
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

    # Convert time to minutes
    t = (t - t.min()).dt.total_seconds() / 60

    CHARGE = data['trip_chrg_pw'].tolist()
    DISCHARGE = data['trip_dischrg_pw'].tolist()

    # Convert energy to kWh
    actual_energy = (np.array(DISCHARGE) - np.array(CHARGE))
    predicted_energy = np.array(model_energy(optimized_params[idx], data))
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

print('Optimized parameters for each file:', optimized_params)

