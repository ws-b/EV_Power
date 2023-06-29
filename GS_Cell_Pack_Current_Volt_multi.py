import os
import pandas as pd
import matplotlib.pyplot as plt

win_folder_path = r'G:\공유 드라이브\Battery Software Lab\Data\한국에너지공과대학교_샘플데이터\trip_by_trip'
mac_folder_path = ''
folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# Create separate lists to store all the data from all the files
all_pack_current = []
all_pack_volt = []
all_cell_volt_list = []
all_t_elapsed_min = []

# Define the figure outside the loop
fig1, ax1 = plt.subplots(figsize=(18, 6))  # for pack_current
fig2, ax2 = plt.subplots(figsize=(18, 6))  # for pack_volt
fig3, ax3 = plt.subplots(figsize=(18, 6))  # for cell_volt_list

# loop over all the files
for file in file_lists[10:80]:
    # create file path
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # Skip files where 'emobility_spd_m_per_s' is always 0
    if data['emobility_spd_m_per_s'].max() == 0:
        continue

    # extract time, pack_current, pack_volt, and the mean cell voltage
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    t_start = t.iloc[0]
    t_elapsed_min = (t - t_start).dt.total_seconds() / 60  # convert time elapsed to minutes
    pack_current = data['pack_current'].tolist()
    pack_volt = data['pack_volt'].tolist()

    # calculate the mean of cell voltages
    cell_volt_list = [item.split(',') for item in data['cell_volt_list']]
    cell_volt_mean = [sum(map(float, item)) / len(item) for item in cell_volt_list]

    # only plot the graph if the time range is more than 1 hour
    time_range = t.iloc[-1] - t_start
    if time_range.total_seconds() >= 3600:  # 1 hour = 3600 seconds
        ax1.plot(t_elapsed_min, pack_current)
        ax2.plot(t_elapsed_min, pack_volt)
        ax3.plot(t_elapsed_min, cell_volt_mean)

# # Configure and display the pack_current plot
# ax1.set_xlabel('Time (minutes)')
# ax1.set_ylabel('Pack Current (A)')
# ax1.set_xlim([0, 60])
# ax1.set_title('Pack Current')
# fig1.show()

# # Configure and display the pack_volt plot
# ax2.set_xlabel('Time (minutes)')
# ax2.set_ylabel('Pack Voltage (V)')
# ax2.set_xlim([0, 60])
# ax2.set_title('Pack Voltage')
# fig2.show()
#
# Configure and display the mean cell voltage plot
ax3.set_xlabel('Time (minutes)')
ax3.set_ylabel('Mean Cell Voltage (V)')
ax3.set_xlim([0, 60])
ax3.set_title('Cell Voltage')
fig3.show()