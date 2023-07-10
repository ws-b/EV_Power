import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import math
from tqdm import tqdm

win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터 속도-가속도 처리'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_avg_speed_and_mileage = []

for file in tqdm(file_lists):
    # Create the file path
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)

    # Extract time, latitude, longitude, speed, acceleration, total distance, and power
    t, lat, log, v, a, total_distance, Power = data.T

    # Calculate the total power
    total_power = np.sum(Power)

    # Calculate the total time
    total_time = np.sum(np.diff(t))

    # Calculate the average speed (km/h)
    avg_speed = np.mean(v) * 3.6

    # Calculate the mileage (km/kWh) for each file (set to 0 if total_power is 0)
    mileage = (total_distance[-1] / 1000) / ((total_power / 1000) * (total_time / 3600)) if total_power != 0 else 0

    # Collect the average speed and mileage values for all files
    all_avg_speed_and_mileage.append((avg_speed, mileage))

# Categorize average speed into 10 km/h intervals
bins = range(0, 160, 10)
binned_avg_speed_and_mileage = [[] for _ in bins]

for avg_speed, mileage in all_avg_speed_and_mileage:
    bin_index = math.floor(avg_speed / 10)
    if bin_index < len(bins):
        binned_avg_speed_and_mileage[bin_index].append(mileage)

# Box plots for each category
plt.figure(figsize=(10, 6))
sns.boxplot(data=binned_avg_speed_and_mileage)
plt.xlabel('Average Speed (km/h)')
plt.ylabel('Mileage (km/kWh)')
plt.xticks(range(len(bins)), bins)

# Limit the x-axis and y-axis ranges
plt.ylim(0, 16)
plt.xlim(-0.5, 10.5)

plt.title('Mileage by Average Speed')
plt.show()
