import plotly.express as px
import pandas as pd

path = '/Users/woojin/Desktop/대학교 자료/켄텍 자료/삼성미래과제/TripEnergy/Travel_dataset1/ChargeCar_Trip_Data/scan_tool_datas/GA/Covington'
file = '/540.txt'
GPS = pd.read_csv(path + file, sep=",", header=None)
GPS.columns = ["GMT time", "relative time", "elevation", "planar distance", "adjusted distance", "speed",
               "acceleration", "power based on model", "Actual Power", "Current", "Voltage"]
""" 
1 lbf = 4.4482 N
1 mile = 1.60934 km
1 m/s = 2.237 mph
"""

inertia = 0.05
g = 9.18  # m/s**2
v = GPS['speed'].to_list()
a = GPS['acceleration'].to_list()
t = GPS['relative time'].to_list()
Power = GPS['power based on model'].to_list()
A_Power = GPS['Actual Power'].to_list()
s0 = 0
for i in range(0, len(v)-1):
    s0 += v[i]*(t[i+1]-t[i])
for i in range(0, len(Power)):
    Power[i] = -Power[i]
for i in range(0, len(Power)):
    A_Power[i] = -A_Power[i]


class Vehicle:
    def __init__(self, mass, load, Ca, Cb, Cc, eff):
        self.mass = mass  # kg
        self.load = load  # kg
        self.Ca = Ca * 4.44822  # CONVERT lbf to N
        self.Cb = Cb * 4.44822 * 2.237  # lbf/mph-> N/mps
        self.Cc = Cc * 4.44822 * (2.237 ** 2)  # lbf/mph**2-> N/mps**2
        self.eff = eff
        self.pl = 741 # CONSTANT POWER LOSS in J/s

Leaf = Vehicle(1640, 140, 30.08, 0.0713, 0.2206, 0.86035)

E = []
P_a = []
P_b = []
P_c = []
P_d = []
P_e = []
Energy = 0
TripEnergy = 0
Actual_Power = 0
for velocity in v:
    P_a.append(Leaf.Ca * velocity)
    P_b.append(Leaf.Cb * velocity * velocity)
    P_c.append(Leaf.Cc * velocity * velocity * velocity)
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
    TripEnergy += Power[i]
    Actual_Power += A_Power[i]
diff = []

for i in range(0, len(v)):
    diff.append(((E[i] - Power[i])**2)**(1/2))

df1 = pd.DataFrame({'Time':t, 'Pe':P_e, 'Pa':P_a, 'Pb':P_b, 'Pc':P_c, 'Pd':P_d})
fig1 = px.area(df1, x="Time", y=df1.columns[1:6], title='Energy consumption term by term', labels={"value" : "Energy(kW)"})
fig1.show()

df2 = pd.DataFrame({'Time':t,'TripEnergy':Power, 'Energy cal by WJ':E, 'Actual Power':A_Power})
fig2 = px.line(df2, x="Time", y=df2.columns[1:6], title='Energy Usage using TripEnergy', labels={"value" : "Energy(kW)"})
fig2.show()

print(s0)
print(Energy)
print(Actual_Power)
print(TripEnergy)