import os
import pandas as pd
from datetime import datetime
import sklearn.cluster as clst
import numpy as np
import matplotlib.pyplot as plt

win_path = ''
mac_path = '/Users/woojin/Desktop/대학교 자료/켄텍 자료/삼성미래과제/TripEnergy/Travel_dataset1/ChargeCar_Trip_Data/'
path = mac_path
pathlists = ['scan_tool_datas','gps_datas']
triplist = []
for pathlist in pathlists:
    merged_path : str = os.path.join(path, pathlist)
    dirs = os.listdir(merged_path)
    dirs = [dir for dir in dirs if not dir.startswith ('.')] # exclude .DS_Store
    dirs = [dir for dir in dirs if not dir.endswith('.txt')] # exclude .txt file
    for dir in dirs:
        dir_path = os.path.join(merged_path, dir)
        for (root, directories, files) in os.walk(dir_path):
            files = [file for file in files if not file.startswith('.')]  # exclude .DS_Store
            files = [file for file in files if '.txt' in file]  # include .txt file only
            for file in files:
                file_path = os.path.join(root, file)
                GPS = pd.read_csv(file_path, sep=",", header=None)
                if 'scan_tool_datas' in dir_path:
                    GPS.columns = ["GMT time", "relative time", "elevation", "planar distance", "adjusted distance", "speed", "acceleration", "power based on model", "Actual Power", "Current", "Voltage"]
                elif 'gps_datas' in dir_path:
                    GPS.columns = ["GMT time", "relative time", "elevation", "planar distance", "adjusted distance", "speed", "acceleration", "power based on model"]
                else:
                    print("PATH ERROR")
                    break
                n = os.path.basename(file_path)[:-4]
                m = 1
                GMT_Time = GPS['GMT time'].tolist()
                cut = [0]
                for i in range(1, len(GMT_Time)):
                    if abs((datetime.strptime(str(int(GMT_Time[i])).zfill(6), "%H%M%S") - datetime.strptime(str(int(GMT_Time[i-1])).zfill(6),"%H%M%S")).total_seconds()) >= 600 : # i where difference of time rows surpass 600 seconds
                        cut.append(i)
                    elif i + 1 == len(GMT_Time):
                        cut.append(i)
                for i in range(1, len(cut)):
                    globals()['Trip' + '_' + str(n).zfill(3) + '_' + str(m).zfill(2)] = GPS.iloc[cut[i-1]:cut[i]]   #Parsing Trip with i series
                    triplist.append([str(n).zfill(3), str(m).zfill(2)])
                    m += 1
                    if m == len(cut):
                        m = 1