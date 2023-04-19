import numpy as np

win_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\Driving Pattern\Drive Cycle' #only for Windows
mac_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터 속도:가속도 처리/'
path = mac_path
file = '164_1.csv'

# Load CSV file into a numpy array
data = np.genfromtxt(path+file, delimiter=',')

""" 
1 lbf = 4.4482 N
1 mile = 1.60934 km
1 m/s = 2.237 mph
"""

inertia = 0.05
t = data[:,0].tolist()
v = data[:,3].tolist()
a = data[:,4].tolist()


class Vehicle:
    def __init__(self, mass, load, Ca, Cb, Cc, eff):
        self.mass = mass  # kg
        self.load = load  # kg
        self.Ca = Ca * 4.44822  # CONVERT lbf to N
        self.Cb = Cb * 4.44822 * 2.237  # lbf/mph-> N/mps
        self.Cc = Cc * 4.44822 * (2.237 ** 2)  # lbf/mph**2-> N/mps**2
        self.eff = eff
        self.pl = 741 # CONSTANT POWER LOSS in J/s

    def getengery(self):
        E = []
        P_a = []
        P_b = []
        P_c = []
        P_d = []
        P_e = []
        Energy = 0
        for i in v:
            P_a.append(Leaf.Ca * i)
            P_b.append(Leaf.Cb * i * i)
            P_c.append(Leaf.Cc * i * i * i)
            P_e.append(Leaf.pl)
        for i in range(0, len(v)):
            if a[i] >= 0:
                P_d.append(((1 + inertia) * (Leaf.mass + Leaf.load) * a[i]) / Leaf.eff)  # BATTERY ENERGY USAGE
            else:
                P_d.append(((1 + inertia) * (Leaf.mass + Leaf.load) * a[i]) * Leaf.eff)
            P_d[i] = P_d[i] * v[i]
            E.append(P_a[i] + P_b[i] + P_c[i] + P_d[i] + P_e[i])
            E[i] = E[i] * 2.77778e-7*3600 # J -> kWh
            Energy += E[i]
        return Energy

Leaf = Vehicle(1520, 0, 30.08, 0.0713, 0.2206, 0.86035)
print(Leaf.getengery())