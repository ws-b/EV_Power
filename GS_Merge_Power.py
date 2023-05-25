import os
import numpy as np
import pandas as pd

# 파일이 들어있는 폴더 경로
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'
mac_folder_path = '/Users/woojin/Documents/켄텍 자료/삼성미래과제/한국에너지공과대학교_샘플데이터/Ioniq5/'
folder_path = mac_folder_path

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()
for file_list in file_lists:
    file = open(folder_path + file_list, "r")

    data = pd.read_csv(file)

    # Set parameters for vehicle model
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
            self.Cb = Cb * 4.44822 * 2.237  # lbf/mph-> N/mps # Rolling resistance coefficient
            self.Cc = Cc * 4.44822 * (2.237 ** 2)  # lbf/mph**2-> N/mps**2 # Gradient resistance coefficient
            self.eff = eff # Efficiency
            self.aux = aux # Auxiliary Power, Not considering Heating and Cooling
            self.idle = idle # IDLE Power

    # Calculate power demand for air resistance, rolling resistance, and gradient resistance
    ioniq5 = Vehicle(2268, 0, 34.342, 0.21928, 0.022718, 737, 100, 0.90)

    Power = []
    P_a = []
    P_b = []
    P_c = []
    P_d = []
    P_e = []

    for velocity in v:
        P_a.append(ioniq5.Ca * velocity / ioniq5.eff / 1000)
        P_b.append(ioniq5.Cb * velocity * velocity/ ioniq5.eff / 1000)
        P_c.append(ioniq5.Cc * velocity * velocity * velocity / ioniq5.eff / 1000)

    # Calculate power demand for acceleration and deceleration
    for i in range(0, len(v)):
        if a[i] >= 0:
            P_d.append(((1 + inertia) * (ioniq5.mass + ioniq5.load) * a[i]) / ioniq5.eff / 1000)  # BATTERY ENERGY USAGE
        else:
            P_d.append((((1 + inertia) * (ioniq5.mass + ioniq5.load) * a[i]) / ioniq5.eff / 1000) + ((1 + inertia) * (ioniq5.mass + ioniq5.load) * abs(a[i]) / np.exp(0.04111 / min(abs(a[i]), 1e10)) / 1000))

        P_d[i] = P_d[i] * v[i]
        if v[i] <= 0.5:
            P_e.append((ioniq5.aux + ioniq5.idle) / 1000)
        else:
            P_e.append(ioniq5.aux / 1000)
        Power.append((P_a[i] + P_b[i] + P_c[i] + P_d[i]+ P_e[i]))

    # Convert Power list to a numpy array and reshape it to match the number of rows in the data array
    Power_array = np.array(Power).reshape(-1, 1)

    # Use numpy.column_stack() to append the Power_array as a new column
    data = np.column_stack((data, Power_array))

    # Export the array to a text file
    save_path = folder_path  # Save to the same folder as the input file


