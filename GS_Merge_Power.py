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
    elif car == 'Ionic5':
        return Vehicle(2268, 0, 34.342, 0.21928, 0.022718, 250, 350, 0, 0.9) # parameters for Ioniq5
    elif car == 'Ionic6':
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

    # Set parameters for the vehicle model
    inertia = 0.05  # rotational inertia of the wheels
    g = 9.18  # m/s**2
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    v = data['speed'].tolist()
    a = data['acceleration'].tolist()
    int_temp = data['int_temp'].tolist()

    A = []
    B = []
    C = []
    D = []
    E = []

    for velocity in v:
        A.append(EV.Ca * velocity / EV.eff)
        B.append(EV.Cb * velocity * velocity / EV.eff)
        C.append(EV.Cc * velocity * velocity * velocity / EV.eff)

    for i in range(len(a)):
        if EV.re_brake == 1:
            if abs(a[i]) < 0.001:  # Threshold for acceleration to avoid division by zero
                exp_term = np.exp(0.0411 / 0.001)  # Use the threshold value instead of actual acceleration
            else:
                exp_term = np.exp(0.0411 / abs(a[i]))

            if a[i] >= 0:
                D.append(((1 + inertia) * (EV.mass + EV.load) * a[i]) * v[i] / EV.eff)
            else:
                D.append((((1 + inertia) * (EV.mass + EV.load) * a[i] * v[i] / exp_term)) * EV.eff)
        else:
            if a[i] >= 0:
                D.append(((1 + inertia) * (EV.mass + EV.load) * a[i]) * v[i] / EV.eff)
            else:
                D.append(0)
        Eff_hvac = 0.81 # Auxiliary power efficiency
        target_int_temp = 22
        E_hvac = abs(target_int_temp - int_temp[i]) * EV.hvac * Eff_hvac # 22'c is the set temperature
        if v[i] <= 0.5:
            E.append(EV.aux + EV.idle + E_hvac)
        else:
            E.append(EV.aux + E_hvac)

    Power_list = [A, B, C, D, E]
    Power = np.sum(Power_list, axis=0)

    data['Power'] = Power

    data.to_csv(file, index=False)

def process_files_power(file_lists, EV):
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_power, file, EV) for file in file_lists]
        for future in tqdm(futures):
            future.result()

    print('Done')