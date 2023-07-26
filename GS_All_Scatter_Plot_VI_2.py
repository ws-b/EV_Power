import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress

win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\ioniq5'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# Create empty lists to store final values
final_energy_data = []
final_net_charge = []

for file in tqdm(file_lists):
    # create file path
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # extract time, energy, CHARGE, DISCHARGE
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    power = data['Power_IV'].tolist()

    # calculate time differences in seconds
    time_diff = t.diff().dt.total_seconds().fillna(0)

    # calculate difference between CHARGE and DISCHARGE
    energy_data = power * time_diff / 3600 / 1000
    energy_data_cumulative = energy_data.cumsum()

    # extract time, energy, CHARGE, DISCHARGE
    CHARGE = data['trip_chrg_pw'].tolist()
    DISCHARGE = data['trip_dischrg_pw'].tolist()

    # calculate difference between CHARGE and DISCHARGE
    net_charge = np.array(DISCHARGE) - np.array(CHARGE)

    final_energy_data.append(energy_data_cumulative.iloc[-1])
    final_net_charge.append(net_charge[-1])

# Add trendline
slope, intercept, r_value, p_value, std_err = linregress(final_net_charge, final_energy_data)

# calculate the residuals (how far is each data point from the trendline)
residuals = np.array(final_energy_data) - (intercept + slope*np.array(final_net_charge))

# calculate the standard deviation of the residuals
residuals_std = np.std(residuals)

# define an "outlier" as any point more than 2 standard deviations from the trendline
outlier_threshold = 2*residuals_std

# find the indices of the outliers
outlier_indices = [i for i, r in enumerate(residuals) if abs(r) > outlier_threshold]

# plot the graph
fig, ax = plt.subplots(figsize=(6, 6))  # set the size of the graph

ax.set_xlabel('Cumulative Energy (kWh)')  # changed
ax.set_ylabel('Energy(data)(kWh)')  # changed

# scatter plot with different colors for outliers and non-outliers
for i, (x, y) in enumerate(zip(final_net_charge, final_energy_data)):
    if i in outlier_indices:
        color = 'tab:red'
    else:
        color = 'tab:blue'
    ax.scatter(x, y, color=color)  # swapped the x and y variables

ax.plot(np.array(final_net_charge), intercept + slope*np.array(final_net_charge), 'b', label='fitted line')

# Create y=x line
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.title('Energy(Voltage x Current) vs. Net Charge over Time')
plt.show()

# find the files corresponding to the outliers
outlier_files = [file_lists[i] for i in outlier_indices]

print(f"Outlier files: {outlier_files}")