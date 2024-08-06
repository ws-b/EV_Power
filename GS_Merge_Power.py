import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class Vehicle:
    def __init__(self, mass, load, Ca, Cb, Cc, aux, hvac, idle, eff, re_brake=1):
        self.mass = mass  # kg # Mass of vehicle
        self.load = load  # kg # Load of vehicle
        self.Ca = Ca * 4.44822  # CONVERT lbf to N # Air resistance coefficient
        self.Cb = Cb * 4.44822 * 2.237  # lbf/mph -> N/mps # Rolling resistance coefficient
        self.Cc = Cc * 4.44822 * (2.237 ** 2)  # lbf/mph**2 -> N/mps**2 # Gradient resistance coefficient
        self.eff = eff  # Efficiency
        self.aux = aux  # Auxiliary Power, Not considering Heating and Cooling
        self.hvac = hvac
        self.idle = idle  # IDLE Power
        self.re_brake = re_brake

def select_vehicle(car):
    if car == 'NiroEV':
        return Vehicle(1928, 0, 32.717, -0.19110, 0.023073, 250, 350, 0, 0.9)
    elif car == 'Ioniq5':
        return Vehicle(2268, 0, 34.342, 0.21928, 0.022718, 250, 350, 0, 0.9) # parameters for Ioniq5
    elif car == 'Ioniq6':
        return Vehicle(2041.168, 0, 23.958, 0.15007, 0.015929, 250, 350, 0, 0.9) # parameters for Ionic6
    elif car == 'KonaEV':
        return Vehicle(1814, 0, 24.859, -0.20036, 0.023656, 250, 350, 0, 0.9) # parameters for Kona_EV
    elif car == 'EV6':
        return Vehicle(2154.564, 0, 36.158, 0.29099, 0.019825, 250, 350, 0 , 0.9) # parameters for EV6
    elif car == 'GV60':
        return Vehicle(2154.564, 0, 23.290, 0.23788, 0.019822, 250, 350, 0, 0.9) # parameters for GV60
    elif car == 'Bongo3EV':
        print("Bongo3EV Cannot calculate power consumption. Please select another vehicle.")
        return None
    elif car == 'Porter2EV':
        print("Porter2EV Cannot calculate power consumption. Please select another vehicle.")
        return None
    else:
        print("Invalid choice. Please try again.")
        return None

def process_file_power(file, EV):
    data = pd.read_csv(file)

    inertia = 0.05  # Rotational inertia of the wheels
    g = 9.18  # Gravitational acceleration (m/s^2)
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    v = data['speed'].to_numpy()
    a = data['acceleration'].to_numpy()
    ext_temp = data['ext_temp'].to_numpy()

    A = EV.Ca * v / EV.eff
    B = EV.Cb * v**2 / EV.eff
    C = EV.Cc * v**3 / EV.eff
    """
    # 고도 데이터가 있는 경우
    if 'altitude' in data.columns:
        altitude = data['altitude'].to_numpy()
        data = data.assign(altitude=data['altitude'].interpolate(method='linear'))

        # 고도 차이 구하기
        altitude_diff = np.diff(altitude)
        altitude_diff = np.append(altitude_diff, 0)  # 마지막 값을 0으로 설정

        # 시간 간격 계산 (초 단위)
        time_diff = np.diff(t.astype(np.int64) // 10 ** 9)
        time_diff = np.append(time_diff, time_diff[-1])  # 마지막 값을 이전 값으로 설정

        # 거리 계산 (속도 * 시간 간격)
        distance_diff = v * time_diff

        # 각도 계산
        slope = np.arctan2(altitude_diff / distance_diff)

        # 힘 계산
        #F = EV.mass * g * np.sin(slope) * v / EV.eff
        F = np.zeros_like(v)
    else:
        F = np.zeros_like(v)
    """
    D = []
    F = np.zeros_like(v)
    for i in range(len(a)):
        if EV.re_brake == 1:
            exp_term = np.exp(0.0411 / max(abs(a[i]), 0.001))
            if a[i] >= 0:
                D.append(((1 + inertia) * (EV.mass + EV.load) * a[i] * v[i]) / EV.eff)
            else:
                D.append((((1 + inertia) * (EV.mass + EV.load) * a[i] * v[i] / exp_term)) * EV.eff)
        else:
            D.append(((1 + inertia) * (EV.mass + EV.load) * a[i] * v[i]) / EV.eff if a[i] >= 0 else 0)

    Eff_hvac = 0.81  # Auxiliary power efficiency
    target_temp = 22
    E_hvac = abs(target_temp - ext_temp) * EV.hvac * Eff_hvac

    E = []
    for i in range(len(v)):
        if v[i] <= 0.5:
            E.append(EV.aux + EV.idle + E_hvac[i])
        else:
            E.append(EV.aux + E_hvac[i])

    Power = np.array(A) + np.array(B) + np.array(C) + np.array(D) + np.array(E) + np.array(F)
    data['Power'] = Power

    data.to_csv(file, index=False)

def process_files_power(file_lists, EV):
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_power, file, EV) for file in file_lists]
        for future in tqdm(futures):
            future.result()

    print('Done')

def calculate_rrmse(y_test, y_pred):
    relative_errors = (y_pred - y_test) / np.mean(y_test)
    rrmse = np.sqrt(np.mean(relative_errors ** 2))
    return rrmse
def read_and_process_files(files):
    data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    return data

def compute_rrmse(vehicle_files, selected_car):
    if not vehicle_files:
        print("No files provided")
        return

    data = read_and_process_files(vehicle_files[selected_car])

    if 'Power' not in data.columns or 'Power_IV' not in data.columns:
        print(f"Columns 'Power' and/or 'Power_IV' not found in the data")
        return

    y_pred = data['Power'].to_numpy()
    y_test = data['Power_IV'].to_numpy()

    rrmse = calculate_rrmse(y_test, y_pred)
    print(f"RRMSE for {selected_car}  : {rrmse}")
    return rrmse