import pandas as pd
from datetime import datetime

path = '/Users/woojin/Desktop/대학교 자료/켄텍 자료/삼성미래과제/TripEnergy/Travel_dataset1/ChargeCar_Trip_Data/gps_datas/CT/Stamford/'
file = '/447.txt'
GPS = pd.read_csv(path + file, sep=",", header=None)
GPS.columns = ["GMT time", "relative time", "elevation", "planar distance", "adjusted distance", "speed",
               "acceleration", "power based on model"]
n = 447
m = 1
GMT_Time = GPS['GMT time'].tolist()
cut = [0]
for i in range(1, len(GMT_Time)): # len(GPS) : number of rows
    if abs((datetime.strptime(str(GMT_Time[i]), "%H%M%S") - datetime.strptime(str(GMT_Time[i-1]),"%H%M%S")).total_seconds()) >= 300 : # i where difference of time rows surpass 300 seconds
        cut.append(i)
    elif i + 1 == len(GMT_Time):
        cut.append(i)
for j in range(1, len(cut)):
    globals()['Trip' + '_' + str(n) + '_' + str(m)] = GPS.iloc[cut[j-1]:cut[j]] #Parsing Trip with i series
    m += 1
    if m == len(cut):
        m = 1