import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

win_folder_path = 'G:\\공유 드라이브\\Battery Software Lab\\Data\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_distance_per_total_power = []
all_average_speed = []

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

    # Calculate the distance per total power (km/kWh) for each file (set to 0 if total_power is 0)
    distance_per_total_power_km_kWh = (total_distance[-1] / 1000) / ((total_power / 1000) * (total_time / 3600)) if total_power != 0 else 0

    # Collect the distance per total power values for all files
    all_distance_per_total_power.append(distance_per_total_power_km_kWh)

    # Calculate the average speed (m/s to km/h) for each file
    all_average_speed.append(np.mean(v) * 3.6)

# Convert the data into a 2D histogram
hist, xedges, yedges = np.histogram2d(all_average_speed, all_distance_per_total_power, bins=30)

# Draw a heatmap
sns.heatmap(hist, cmap='viridis', xticklabels=5, yticklabels=5)

# Set the axis labels
plt.xlabel('Average Speed (km/h)')
plt.ylabel('Total Distance / Total Power (km/kWh)')

# Set the title of the plot
plt.title('Heatmap of Average Speed vs Electric Mileage')

# Display grid
plt.grid(False)

# Limit the x-axis and y-axis ranges (up to 40)
plt.xlim(0, 40)
plt.ylim(0, 10)

# Show the plot
plt.show()
