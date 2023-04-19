import os
import numpy as np

# 파일이 들어있는 폴더 경로
win_folder_path =''
mac_folder_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터 속도:가속도 처리/'
folder_path = mac_folder_path

def get_file_list(folder_path):
    # 폴더 내의 모든 파일 리스트 가져오기
    file_list = os.listdir(folder_path)
    txt_files = []
    for file in file_list:
        if file.endswith('.csv'):
            txt_files.append(file)
    return txt_files

# 파일 리스트 가져오기
files = get_file_list(folder_path)
files.sort()

for file in files:
    # 파일 경로 생성하기
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)

    # Set parameters for vehicle model
    inertia = 0.05
    g = 9.18  # m/s**2
    t = data[:,0].tolist()
    t = [int(x) for x in t]
    v = data[:,3].tolist()
    a = data[:,4].tolist()

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
    model3 = Vehicle(1930, 0, 38.510, -0.08110, 0.016100, 737, 100, 0.87)

    Power = []
    P_a = []
    P_b = []
    P_c = []
    P_d = []
    P_e = []

    for velocity in v:
        P_a.append(model3.Ca * velocity / model3.eff / 1000)
        P_b.append(model3.Cb * velocity * velocity/ model3.eff / 1000)
        P_c.append(model3.Cc * velocity * velocity * velocity / model3.eff / 1000)

    # Calculate power demand for acceleration and deceleration
    for i in range(0, len(v)):
        if a[i] >= 0:
            P_d.append(((1 + inertia) * (model3.mass + model3.load) * a[i]) / model3.eff / 1000)  # BATTERY ENERGY USAGE
        else:
            P_d.append((((1 + inertia) * (model3.mass + model3.load) * a[i]) / model3.eff / 1000) + ((1 + inertia) * (model3.mass + model3.load) * abs(a[i]) / np.exp(0.04111/abs(a[i])) / 1000))
        P_d[i] = P_d[i] * v[i]
        if v[i] <= 0.5:
            P_e.append((model3.aux + model3.idle) / 1000)
        else:
            P_e.append(model3.aux / 1000)
        Power.append((P_a[i] + P_b[i] + P_c[i] + P_d[i]+ P_e[i]))

