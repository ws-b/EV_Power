import os
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\ioniq5'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# Create empty lists to store final values
final_net_charge = []
final_energy = []

for file in tqdm(file_lists):
    # create file path
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # extract time, energy, CHARGE, DISCHARGE
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    CHARGE = data['trip_chrg_pw'].tolist()
    DISCHARGE = data['trip_dischrg_pw'].tolist()

    # calculate difference between CHARGE and DISCHARGE
    net_charge = np.array(DISCHARGE) - np.array(CHARGE)

    # convert energy data to kWh and perform cumulative calculation
    energy_kWh = data['Energy']
    energy_kWh_cumulative = energy_kWh.cumsum()
    final_net_charge.append(net_charge[-1])
    final_energy.append(energy_kWh_cumulative.iloc[-1])

# plot the graph
fig, ax = plt.subplots(figsize=(6, 6))  # set the size of the graph

ax.set_xlabel('Cumulative Energy (kWh)')  # changed
ax.set_ylabel('Net Charge (Discharge - Charge) (kWh)')  # changed
ax.scatter(final_energy, final_net_charge, color='tab:blue')  # swapped the x and y variables

# Create y=x line
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.title('Net Charge vs. Cumulative Energy over Time')
plt.show()
