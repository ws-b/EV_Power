import os
import numpy as np
import matplotlib.pyplot as plt

# Folder paths
win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터 속도-가속도 처리'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# Choose a file
file = file_lists[35]

# Create the file path
file_path = os.path.join(folder_path, file)

# Load data from the CSV file
data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)

# Set parameters for the vehicle model
inertia = 0.05
g = 9.18  # m/s**2
t = data[:, 0].tolist()
t = [int(x) for x in t]
v = data[:, 3].tolist()
a = data[:, 4].tolist()

class Vehicle:
    def __init__(self, mass, load, Ca, Cb, Cc, aux, idle, eff):
        self.mass = mass  # kg # Mass of the vehicle
        self.load = load  # kg # Load of the vehicle
        self.Ca = Ca * 4.44822  # CONVERT lbf to N # Air resistance coefficient
        self.Cb = Cb * 4.44822 * 2.237  # lbf/mph -> N/mps # Rolling resistance coefficient
        self.Cc = Cc * 4.44822 * (2.237 ** 2)  # lbf/mph**2 -> N/mps**2 # Gradient resistance coefficient
        self.eff = eff  # Efficiency
        self.aux = aux  # Auxiliary Power, Not considering Heating and Cooling
        self.idle = idle  # IDLE Power

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
    P_b.append(model3.Cb * velocity * velocity / model3.eff / 1000)
    P_c.append(model3.Cc * velocity * velocity * velocity / model3.eff / 1000)

# Calculate power demand for acceleration and deceleration
for i in range(0, len(v)):
    if a[i] >= 0:
        P_d.append(((1 + inertia) * (model3.mass + model3.load) * a[i]) / model3.eff / 1000)  # BATTERY ENERGY USAGE
    else:
        P_d.append((((1 + inertia) * (model3.mass + model3.load) * a[i]) / model3.eff / 1000) + ((1 + inertia) * (model3.mass + model3.load) * abs(a[i]) / np.exp(0.04111 / abs(a[i])) / 1000))

    P_d[i] = P_d[i] * v[i]
    if v[i] <= 0.5:
        P_e.append((model3.aux + model3.idle) / 1000)
    else:
        P_e.append(model3.aux / 1000)
    Power.append((P_a[i] + P_b[i] + P_c[i] + P_d[i] + P_e[i]))

# Plotting
plt.figure(figsize=(10, 6))

# Convert seconds to minutes
t_minutes = [i / 60 for i in t]

plt.stackplot(t_minutes, P_a, P_b, P_c, P_d, P_e, labels=['P_a', 'P_b', 'P_c', 'P_d', 'P_e'])
plt.legend(loc='upper left')
plt.title('Stacked Plot of Power')
plt.xlabel('Time (minutes)')
plt.ylabel('Power')
plt.show()
