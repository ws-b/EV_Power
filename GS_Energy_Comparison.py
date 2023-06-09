import os
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\ioniq5'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# plot graphs for each file
for file in tqdm(file_lists[25:29]):
    # create file path
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # extract time, Power, CHARGE, DISCHARGE
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    CHARGE = data['trip_chrg_pw'].tolist()
    DISCHARGE = data['trip_dischrg_pw'].tolist()

    # calculate difference between CHARGE and DISCHARGE
    net_charge = np.array(DISCHARGE) - np.array(CHARGE)

    # convert Power data to kWh and perform cumulative calculation
    Power_kWh = data['Energy']  # convert kW to kWh considering the 2-second time interval
    Power_kWh_cumulative = Power_kWh.cumsum()

    # only plot the graph if the time range is more than 5 minutes
    time_range = t.iloc[-1] - t.iloc[0]
    if time_range.total_seconds() >= 300 and Power_kWh_cumulative.iloc[-1] >= 1:    # 5 minutes = 300 seconds

        # plot the graph
        plt.figure(figsize=(10, 6))  # set the size of the graph

        plt.xlabel('Time')
        plt.ylabel('Cumulative Power and Net Charge (kWh)')
        plt.plot(t, Power_kWh_cumulative, label='Cumulative Power (kWh)', color='tab:blue')
        plt.plot(t, net_charge, label='Net Charge (kWh)', color='tab:red')

        # format the ticks
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # change to display only time
        plt.gca().set_xticks([t.iloc[0], t.iloc[-1]])  # set x-axis ticks to only start and end

        # add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))  # Moving legend slightly down
        plt.title('Cumulative Energy (kWh) and Net Charge')
        plt.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
