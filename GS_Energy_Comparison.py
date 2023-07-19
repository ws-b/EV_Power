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

# Select a random subset of 10 test files
random_test_files = np.random.choice(file_lists, 10)

# Select a random subset of 10 test files
random_test_files = np.random.choice(file_lists, 10)

# plot graphs for each file
#for file in tqdm(random_test_files):
for file in tqdm(file_lists[25:30]):
    # create file path
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # extract time, Power, CHARGE, DISCHARGE
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    t_diff = (t - t.iloc[0]).dt.total_seconds() / 60  # convert time difference to minutes
    Energy_VI = data['Energy_VI'].tolist()

    # convert Power data to kWh
    Power_kWh = data['Energy']  # convert kW to kWh considering the 2-second time interval

    # calculate the difference between the energies
    energy_diff = Power_kWh - Energy_VI

    # plot the comparison graph
    plt.figure(figsize=(10, 6))  # set the size of the graph
    plt.xlabel('Time (minutes)')
    plt.ylabel('BMS Energy and Model Energy (kWh)')
    plt.plot(t_diff, Power_kWh, label='Model Energy (kWh)', color='tab:blue')
    plt.plot(t_diff, Energy_VI, label='BMS Energy (kWh)', color='tab:red')

    # add date and file name
    date = t.iloc[0].strftime('%Y-%m-%d')
    plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right', color='black')
    plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='left', color='black')


    plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))  # Moving legend slightly down
    plt.title('Model Energy vs. BMS Energy')
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # # plot the difference graph
    # plt.figure(figsize=(10, 6))  # set the size of the graph
    # plt.xlabel('Time (minutes)')
    # plt.ylabel('Difference between BMS Energy and Model Energy (kWh)')
    # plt.plot(t_diff, energy_diff, label='Difference (kWh)', color='tab:blue')
    #
    # # add date and file name
    # date = t.iloc[0].strftime('%Y-%m-%d')
    # plt.text(1, 1, date, transform=plt.gca().transAxes, fontsize=12,
    #          verticalalignment='top', horizontalalignment='right', color='black')
    # plt.text(0, 1, 'File: '+file, transform=plt.gca().transAxes, fontsize=12,
    #          verticalalignment='top', horizontalalignment='left', color='black')
    #
    #
    # plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))  # Moving legend slightly down
    # plt.title('Difference between Model Energy and BMS Energy')
    # plt.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()
