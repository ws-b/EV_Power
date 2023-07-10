import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

win_folder_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\trip_by_trip'
mac_folder_path = ''
folder_path = os.path.normpath(win_folder_path)

folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

for file in tqdm(file_lists[20:29]):
    # create file path
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # extract time, pack_current, pack_volt, and the mean cell voltage
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    t_start = t.iloc[0]
    t_elapsed_min = (t - t_start).dt.total_seconds() / 60  # convert time elapsed to minutes
    pack_current = data['pack_current'].tolist()
    pack_volt = data['pack_volt'].tolist()

    # calculate the mean of cell voltages
    cell_volt_list = [item.split(',') for item in data['cell_volt_list']]
    cell_volt_mean = [sum(map(float, item))/len(item) for item in cell_volt_list]

    # only plot the graph if the time range is more than 1 hour
    time_range = t.iloc[-1] - t_start
    if time_range.total_seconds() >= 3600:  # 1 hour = 3600 seconds
        # plot the graph for pack_current
        fig, ax1 = plt.subplots(figsize=(18, 6))  # set the size of the graph

        color = 'tab:blue'
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Pack Current (A)', color=color)
        ax1.plot(t_elapsed_min, pack_current, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.subplots_adjust(top=0.9)  # adjust the top of the subplot

        # # add date and file name
        # date = t.iloc[0].strftime('%Y-%m-%d')
        # plt.text(1, 1, date, transform=ax1.transAxes, fontsize=12,
        #          verticalalignment='top', horizontalalignment='right', color='black')
        # plt.text(0, 1, 'File: '+file, transform=ax1.transAxes, fontsize=12,
        #          verticalalignment='top', horizontalalignment='left', color='black')

        plt.title('Pack Current')
        plt.show()

        # plot the graph for pack_volt
        fig, ax1 = plt.subplots(figsize=(18, 6))  # set the size of the graph

        color = 'tab:red'
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Pack Voltage (V)', color=color)
        ax1.plot(t_elapsed_min, pack_volt, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.subplots_adjust(top=0.9)  # adjust the top of the subplot

        # # add date and file name
        # date = t.iloc[0].strftime('%Y-%m-%d')
        # plt.text(1, 1, date, transform=ax1.transAxes, fontsize=12,
        #          verticalalignment='top', horizontalalignment='right', color='black')
        # plt.text(0, 1, 'File: '+file, transform=ax1.transAxes, fontsize=12,
        #          verticalalignment='top', horizontalalignment='left', color='black')

        plt.title('Pack Voltage')
        plt.show()

        # plot the graph for the mean cell voltage
        fig, ax1 = plt.subplots(figsize=(18, 6))  # set the size of the graph

        color = 'tab:green'
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Cell Voltage (V)', color=color)
        ax1.plot(t_elapsed_min, cell_volt_mean, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.subplots_adjust(top=0.9)  # adjust the top of the subplot

        # # add date and file name
        # date = t.iloc[0].strftime('%Y-%m-%d')
        # plt.text(1, 1, date, transform=ax1.transAxes, fontsize=12,
        #          verticalalignment='top', horizontalalignment='right', color='black')
        # plt.text(0, 1, 'File: '+file, transform=ax1.transAxes, fontsize=12,
        #          verticalalignment='top', horizontalalignment='left', color='black')

        plt.title('Cell Voltage')
        plt.show()