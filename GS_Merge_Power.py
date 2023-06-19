import os
import numpy as np
import pandas as pd

# folder path where the files are stored
win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\한국에너지공과대학교_샘플데이터\ioniq5'
folder_path = os.path.normpath(win_folder_path)

"""
차종별 차량번호
포터2 - 01241228177
코나EV - 01241248726
아이오닉5 - 01241248782
"""

# Get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# Iterate over each file
for file in file_lists:
    file_path = os.path.join(folder_path, file)

    # Read the data from the file into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Set parameters for the vehicle model
    inertia = 0.05
    g = 9.18  # m/s**2
    t = data['time'].tolist()
    v = data['emobility_spd_m_per_s'].tolist()
    a = data['acceleration'].tolist()

    class Vehicle:
        def __init__(self, mass, load, Ca, Cb, Cc, aux, idle, eff):
            self.mass = mass  # kg # Mass of vehicle
            self.load = load  # kg # Load of vehicle
            self.Ca = Ca * 4.44822  # CONVERT lbf to N # Air resistance coefficient
            self.Cb = Cb * 4.44822 * 2.237  # lbf/mph -> N/mps # Rolling resistance coefficient
            self.Cc = Cc * 4.44822 * (2.237 ** 2)  # lbf/mph**2 -> N/mps**2 # Gradient resistance coefficient
            self.eff = eff  # Efficiency
            self.aux = aux  # Auxiliary Power, Not considering Heating and Cooling
            self.idle = idle  # IDLE Power

    # Calculate power demand for air resistance, rolling resistance, and gradient resistance
    ioniq5 = Vehicle(2268, 0, 34.342, 0.21928, 0.022718, 737, 100, 0.87)
    kona_ev = Vehicle(1814, 0, 24.859, -0.20036, 0.023656, 737, 100, 0.87)
    EV = ioniq5

    Power = []
    P_a = []
    P_b = []
    P_c = []
    P_d = []
    P_e = []

    for velocity in v:
        P_a.append(EV.Ca * velocity / EV.eff / 1000)
        P_b.append(EV.Cb * velocity * velocity / EV.eff / 1000)
        P_c.append(EV.Cc * velocity * velocity * velocity / EV.eff / 1000)

    # Calculate power demand for acceleration and deceleration
    for i in range(0, len(v)):
        if a[i] >= 0:
            P_d.append(((1 + inertia) * (EV.mass + EV.load) * a[i]) / EV.eff / 1000)  # BATTERY ENERGY USAGE
        else:
            P_d.append((((1 + inertia) * (EV.mass + EV.load) * a[i]) / EV.eff / 1000) + (
                    (1 + inertia) * (EV.mass + EV.load) * abs(a[i]) / np.exp(0.04111 / min(abs(a[i]), 1e10)) / 1000))

        P_d[i] = P_d[i] * v[i]
        if v[i] <= 0.5:
            P_e.append((EV.aux + EV.idle) / 1000)
        else:
            P_e.append(EV.aux / 1000)
        Power.append((P_a[i] + P_b[i] + P_c[i] + P_d[i] + P_e[i]))

    # Add the 'Power' column to the DataFrame
    data['Power'] = Power

    # Overwrite the data to the same .csv file
    data.to_csv(os.path.join(folder_path, file), index=False)