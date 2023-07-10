import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터 속도-가속도 처리'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_total_distance = []
all_total_time = []

for file in tqdm(file_lists):
    # Create the file path
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)

    # Extract time, latitude, longitude, speed, acceleration, total distance, and power
    t, lat, log, v, a, total_distance, Power = data.T

    # Save the total distance and total time for each file
    all_total_distance.append(total_distance[-1])
    all_total_time.append(t[-1])

# Histogram of total distance
plt.figure()
sns.histplot(all_total_distance, bins='auto', color='blue', kde=False)
plt.xlabel('Total Distance (m)')
plt.ylabel('Number of Trips')
plt.title('Total Distance Distribution')
plt.xlim(0, 100000)
plt.grid(False)
plt.show()

# Histogram of total time
plt.figure()
sns.histplot(all_total_time, bins='auto', color='green', kde=False)
plt.xlabel('Total Time (s)')
plt.ylabel('Number of Trips')
plt.title('Total Time Distribution')
plt.xlim(0, 8000)
plt.grid(False)
plt.show()
