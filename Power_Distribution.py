import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Folder path containing the files
win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터 속도-가속도 처리'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# Get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_distance_per_total_power = []
over_50_files = []

for file in file_lists:
    # Create the file path
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)

    # Extract time, latitude, longitude, speed, acceleration, total distance, and power
    t, lat, log, v, a, total_distance, Power = data.T

    # Calculate the total power
    total_power = np.sum(Power)

    # Calculate the total time
    total_time = np.sum(np.diff(t))

    # Calculate Total distance / Total Power for each file (if total_power is 0, set the value to 0)
    distance_per_total_power_km_kWh = (total_distance[-1] / 1000) / ((total_power / 1000) * (total_time / 3600)) if total_power != 0 else 0

    # If distance_per_total_power_km_kWh is greater than or equal to 50, add the file name to over_50_files
    if distance_per_total_power_km_kWh >= 50:
        over_50_files.append(file)
    # Collect all distance_per_total_power values
    all_distance_per_total_power.append(distance_per_total_power_km_kWh)

# Create a histogram for all files
hist_data = sns.histplot(all_distance_per_total_power, bins='auto', color='gray', kde=False)

# Draw a vertical line for the mean
mean_value = np.mean(all_distance_per_total_power)
plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')

# Display the mean value
plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red', fontsize=12)

# # Calculate the mode
# counts, bin_edges = np.histogram(all_distance_per_total_power, bins='auto')
# mode_index = np.argmax(counts)
# mode_value = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2
#
# # Draw a vertical line for the mode
# plt.axvline(mode_value, color='blue', linestyle='--', label=f'Mode: {mode_value:.2f}')
#
# # Display the mode value
# plt.text(mode_value + 0.05, plt.gca().get_ylim()[1] * 0.8, f'Mode: {mode_value:.2f}', color='blue', fontsize=12)

# Set the x-axis range (from 0 to 25)
#plt.xlim(0, 25)
plt.xlabel('Total Distance / Total Power (km/kWh)')
plt.ylabel('Number of trips')
plt.title('Total Distance / Total Power Distribution')
plt.grid(False)
plt.show()

# Print files with a ratio of Total Distance / Total Power greater than 50
print("Files with a ratio of Total Distance / Total Power greater than 50:")
for over_50_file in over_50_files:
    print(over_50_file)
