import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Folder paths
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'
mac_folder_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터/'

folder_path = win_folder_path

def get_file_list(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
    csv_files = []
    for file in file_list:
        if file.endswith('.csv'):
            csv_files.append(file)
    return csv_files

# Get the list of files
files = get_file_list(folder_path)
files.sort()

all_distance_per_total_power = []
all_average_speed = []

for file in files:
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
