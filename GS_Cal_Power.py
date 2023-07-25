import os
import pandas as pd
import numpy as np
from tqdm import tqdm

class Vehicle:
    def __init__(self, mass, load, Ca, Cb, Cc, aux, hvac, idle, eff):
        self.mass = mass  # kg # Mass of vehicle
        self.load = load  # kg # Load of vehicle
        self.Ca = Ca * 4.44822  # CONVERT lbf to N # Air resistance coefficient
        self.Cb = Cb * 4.44822 * 2.237  # lbf/mph -> N/mps # Rolling resistance coefficient
        self.Cc = Cc * 4.44822 * (2.237 ** 2)  # lbf/mph**2 -> N/mps**2 # Gradient resistance coefficient
        self.eff = eff  # Efficiency
        self.aux = aux  # Auxiliary Power, Not considering Heating and Cooling
        self.hvac = hvac
        self.idle = idle  # IDLE Power

def select_vehicle(car):
    if car == 1:
        return Vehicle(2268, 0, 34.342, 0.21928, 0.022718, 250, 350, 0, 0.9) # parameters for Ioniq5
    elif car == 2:
        return Vehicle(1814, 0, 24.859, -0.20036, 0.023656, 250, 350, 0, 0.9) # parameters for Kona_EV
    elif car == 3:
        return Vehicle(0, 0, 0, 0, 0, 0, 0, 0, 0) # parameters for Porter_EV
    else:
        print("Invalid choice. Please try again.")
        return None

def process_files_energy(file_lists, folder_path, EV):
    # Iterate over each file
    for file in tqdm(file_lists):
        file_path = os.path.join(folder_path, file)

        # Read the data from the file into a pandas DataFrame
        data = pd.read_csv(file_path)

        # Set parameters for the vehicle model
        inertia = 0.05 # rotational inertia of the wheels
        g = 9.18  # m/s**2
        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        v = data['speed'].tolist()
        a = data['acceleration'].tolist()
        int_temp = data['int_temp'].tolist()


        F = []  # Force
        Power = []  # Power
        Energy = []  # Energy

        for velocity in v:
            F.append(EV.Ca + EV.Cb * velocity + EV.Cc * velocity * velocity)

        # Calculate power demand for acceleration and deceleration
        for i in range(len(a)):
            F[i] += ((1 + inertia) * (EV.mass + EV.load) * a[i])  # BATTERY ENERGY USAGE
            Power.append(F[i] * v[i] / EV.eff)
            if v[i] <= 0.5:
                Power[i] += (EV.aux + EV.idle)
            else:
                Power[i] += EV.aux
            E_hvac = (22 - int_temp[i]) * EV.hvac # 22'c is the set temperature
            Power[i] += E_hvac

        # Calculate time difference in seconds
        t_diff = t.diff().dt.total_seconds()

        # Convert lists to numpy arrays for vectorized operations
        Power = np.array(Power)
        t_diff = np.array(t_diff.fillna(0))

        # Calculate energy by multiplying power with time difference
        # Convert power from watts to kilowatts and time from seconds to hours
        Energy = Power * t_diff / 3600 / 1000

        # Convert the energy back to a list and add it to the DataFrame
        data['Energy'] = Energy.tolist()

        # Overwrite the data to the same .csv file
        data.to_csv(os.path.join(folder_path, file), index=False)

    print('Done')