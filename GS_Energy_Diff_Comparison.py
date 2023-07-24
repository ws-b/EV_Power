import os
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\ioniq5'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

for file in file_lists[20:30]:
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
    Power_kWh = data['Power'] * 0.00055556  # convert kW to kWh considering the 2-second time interval
    Power_kWh_cumulative = Power_kWh.cumsum()

    # calculate the difference
    diff = Power_kWh_cumulative - net_charge

    # only plot the graph if the time range is more than 5 minutes
    time_range = t.iloc[-1] - t.iloc[0]
    if time_range.total_seconds() >= 300:  # 5 minutes = 300 seconds
        # plot the graph
        fig, ax = plt.subplots(figsize=(10, 6))  # set the size of the graph

        color = 'tab:blue'
        ax.set_xlabel('Time')
        ax.set_ylabel('Difference between Cumulative Power and Net Charge (kWh)', color=color)
        ax.plot(t, diff, color=color)
        ax.tick_params(axis='y', labelcolor=color)

        # format the ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # change to display only time
        ax.set_xticks([t.iloc[0], t.iloc[-1]])  # set x-axis ticks to only start and end

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.subplots_adjust(top=0.9)  # adjust the top of the subplot

        # add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: '+file, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        # set y limit based on the range of diff
        min_val = min(diff)
        max_val = max(diff)
        ax.set_ylim(min_val, max_val)

        plt.title('Difference between Cumulative Power and Net Charge over Time')
        plt.show()
