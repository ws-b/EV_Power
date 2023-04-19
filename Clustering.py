import plotly.express as px
import os
import pandas as pd
from datetime import datetime

path = '/Users/woojin/Desktop/대학교 자료/켄텍 자료/삼성미래과제/TripEnergy/Travel_dataset1/ChargeCar_Trip_Data/'
pathlists = ['scan_tool_datas','gps_datas']

def maketriplist(path, pathlists):
    global triplist
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
                    if 'scan_tool_datas' in dir_path: # scan_tool_datas (EV)
                        GPS.columns = ["GMT time", "relative time", "elevation", "planar distance", "adjusted distance", "speed", "acceleration", "power based on model", "Actual Power", "Current", "Voltage"]
                    elif 'gps_datas' in dir_path: # gps_datas (ICEV)
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
                        globals()['Trip' + '_' + str(n).zfill(3) + '_' + str(m).zfill(2)] = GPS.iloc[cut[i-1]:cut[i]]   # Parsing Trip with i series
                        triplist.append([str(n).zfill(3), str(m).zfill(2)])
                        m += 1
                        if m == len(cut):
                            m = 1
    triplist.sort()

def relative_time():
   for trip in triplist:
      gmttime = globals()['Trip' + '_' + str(trip[0]) + '_' + str(trip[1])]['GMT time'].tolist()
      Times = []
      for i in range(0, len(gmttime)):
         t = (datetime.strptime(str(int(gmttime[i])).zfill(6), "%H%M%S") - datetime.strptime(str(int(gmttime[0])).zfill(6), "%H%M%S")).total_seconds()
         Times.append(t)
      globals()['Trip' + '_' + str(trip[0]) + '_' + str(trip[1])]['relative time'] = Times

maketriplist(path, pathlists)
relative_time()

def plotting():
    v_mean = []
    v_std = []
    a_mean = []
    a_std = []
    time = []
    for trip in triplist:
        v_mean.append(globals()['Trip' + '_' + str(trip[0]) + '_' + str(trip[1])]['speed'].mean())
        v_std.append(globals()['Trip' + '_' + str(trip[0]) + '_' + str(trip[1])]['speed'].std())
        a_mean.append(globals()['Trip' + '_' + str(trip[0]) + '_' + str(trip[1])]['acceleration'].mean())
        a_std.append(globals()['Trip' + '_' + str(trip[0]) + '_' + str(trip[1])]['acceleration'].std())
        time.append(globals()['Trip' + '_' + str(trip[0]) + '_' + str(trip[1])]['relative time'].iloc[-1])
    df = pd.DataFrame({'a_mean': a_mean, 'v_mean': v_mean})
    fig = px.scatter(df, x="a_mean", y="v_mean", labels={"a_mean":"mean accleration (m/s^2)", "v_mean":"mean velocity (m/s)"},width=800, height=600)
    fig.show()

def stop_time():
    global stoptime
    stoptime = []
    for trip in triplist:
        rltime = globals()['Trip' + '_' + str(trip[0]) + '_' + str(trip[1])]['relative time'].tolist()
        Times = 0
        speed = globals()['Trip' + '_' + str(trip[0]) + '_' + str(trip[1])]['speed'].tolist()
        speeds = []
        cut_speed = []
        for i in range(0, len(speed)):
            if speed[i] <= 1.389:
                speeds.append(i)
        for i in range(0, len(speeds)):
            if i == 0:
                if speeds[0] <= 1.389:
                    cut_speed.append(0)
            elif abs(speeds[i]-speeds[i-1]) > 1:
                cut_speed.append(speeds[i-1])
                cut_speed.append(speeds[i])
        if len(cut_speed) % 2 == 1:
            del cut_speed[-1]

        for i in range(0, len(cut_speed)):
            Times += rltime[cut_speed[i]]*(-1)**(i+1)
        stoptime.append(Times)

stop_time()