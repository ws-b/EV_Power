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
    try:
        data = pd.read_csv(file)

        inertia = 0.05  # Rotational inertia of the wheels
        g = 9.18  # Gravitational acceleration (m/s^2)

        # 시간 데이터를 datetime 형식으로 변환
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        data = data.sort_values('time').reset_index(drop=True)

        # 속도, 가속도, 외부 온도 데이터를 numpy 배열로 변환
        v = data['speed'].to_numpy()  # 속도 (m/s assumed)
        a = data['acceleration'].to_numpy()  # 가속도 (m/s^2 assumed)
        ext_temp = data['ext_temp'].to_numpy()

        # 파워 계산을 위한 항목들
        A = EV.Ca * v / EV.eff
        B = EV.Cb * v ** 2 / EV.eff
        C = EV.Cc * v ** 3 / EV.eff

        # 가속도 기반 지수 항 계산
        exp_term = np.exp(0.0411 / np.maximum(np.abs(a), 0.001))

        # 가속도에 따른 D 항 계산
        D_positive = ((1 + inertia) * (EV.mass + EV.load) * a * v) / EV.eff
        D_negative = (((1 + inertia) * (EV.mass + EV.load) * a * v) / exp_term) * EV.eff
        D = np.where(a >= 0, D_positive, np.where(EV.re_brake == 1, D_negative, 0))

        # HVAC 및 보조 전력 계산
        Eff_hvac = 0.81  # Auxiliary power efficiency
        target_temp = 22  # 목표 온도 (°C)
        E_hvac = np.abs(target_temp - ext_temp) * EV.hvac * Eff_hvac
        E = np.where(v <= 0.5, EV.aux + EV.idle + E_hvac, EV.aux + E_hvac)

        # 고도 데이터 처리
        if 'altitude' in data.columns and 'NONE' in data.columns:
            # 고도 데이터를 숫자형으로 변환
            data['altitude'] = pd.to_numeric(data['altitude'], errors='coerce')

            # 결측치 처리 없이 선형 보간 수행
            data['altitude'] = data['altitude'].interpolate(method='linear', limit_direction='both')

            altitude = data['altitude'].to_numpy()

            # 고도 차이 계산 (현재와 이전 고도 차이)
            altitude_diff = np.diff(altitude, prepend=altitude[0])

            # 시간 차이 계산 (초 단위)
            time_diff = data['time'].diff().dt.total_seconds().fillna(2).to_numpy()

            # 거리 차이 계산 (m)
            distance_diff = v * time_diff

            # 경사각 계산 (라디안 단위)
            with np.errstate(divide='ignore', invalid='ignore'):
                slope = np.arctan2(altitude_diff, distance_diff)
                slope = np.where(distance_diff == 0, 0, slope)  # 거리 차이가 0인 경우 경사각을 0으로 설정
            data['slope'] = slope
            # 경사 저항 항 계산
            F = EV.mass * g * np.sin(slope) * v / EV.eff

        else:
            F = np.zeros_like(v)

        # data['A'] = A
        # data['B'] = B
        # data['C'] = C
        # data['D'] = D
        # data['E'] = E
        # data['F'] = F
        data['Power_phys'] =  A + B + C + D + E + F
        # data = data.drop(columns=['A', 'B', 'C', 'D', 'E', 'F'])
        data.to_csv(file, index=False)

    except Exception as e:
        print(f"Error processing file {file}: {e}")

def process_files_power(file_lists, EV):
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_power, file, EV) for file in file_lists]
        for future in tqdm(futures):
            future.result()

    print('Done')