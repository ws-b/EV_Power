import os
import numpy as np
import pandas as pd
from tqdm import tqdm

win_folder_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터_후처리\230710'

folder_path = os.path.normpath(win_folder_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

for file in tqdm(file_lists):
    # Create the file path
    file_path = os.path.join(folder_path, file)

    # Read the data from the file into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Set parameters for the vehicle model
    inertia = 0.05
    g = 9.18  # m/s**2
    t = data['time'].tolist()
    v = data['speed'].tolist()
    a = data['acceleration'].tolist()


    class Vehicle:
        def __init__(self, mass, load, Ca, Cb, Cc, aux, idle, eff):
            self.mass = mass  # kg # Mass of vehicle
            self.load = load  # kg # Load of vehicle
            self.Ca = Ca * 4.44822  # CONVERT lbf to N # Air resistance coefficient
            self.Cb = Cb * 4.44822 * 2.237  # lbf/mph-> N/mps # Rolling resistance coefficient
            self.Cc = Cc * 4.44822 * (2.237 ** 2)  # lbf/mph**2-> N/mps**2 # Gradient resistance coefficient
            self.eff = eff  # Efficiency
            self.aux = aux  # Auxiliary Power, Not considering Heating and Cooling
            self.idle = idle  # IDLE Power

    # Calculate power demand for air resistance, rolling resistance, and gradient resistance
    ioniq5 = Vehicle(2268, 0, 34.342, 0.21928, 0.022718, 870, 100, 0.9)
    kona_ev = Vehicle(1814, 0, 24.859, -0.20036, 0.023656, 870, 100, 0.9)
    EV = ioniq5

    F = []  # Force
    E = []  # Energy

    for velocity in v:
        F.append(EV.Ca + EV.Cb * velocity + EV.Cc * velocity * velocity)

    # Calculate power demand for acceleration and deceleration
    for i in range(len(a)):
        if a[i] >= -0.000001:
            F[i] += ((1 + inertia) * (EV.mass + EV.load) * a[i])  # BATTERY ENERGY USAGE
            E.append(F[i] * v[i] / EV.eff)
        else:
            F[i] += ((((1 + inertia) * (EV.mass + EV.load) * a[i])) + (
                        (1 + inertia) * (EV.mass + EV.load) * abs(a[i]) / np.exp(0.04111 / abs(a[i]))))
            E.append(F[i] * v[i] / EV.eff)

        if v[i] <= 0.5:
            E[i] += (EV.aux + EV.idle)
        else:
            E[i] += EV.aux

    E = [i / 3600000 for i in E]
    # Add the 'Power' column to the DataFrame
    data['Energy'] = E

    # Overwrite the data to the same .csv file
    data.to_csv(os.path.join(folder_path, file), index=False)
print("All files have been processed successfully!")