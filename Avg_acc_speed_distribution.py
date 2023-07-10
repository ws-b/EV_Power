import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

win_folder_path = 'G:\\공유 드라이브\\Battery Software Lab\Data\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_average_acceleration = []
all_average_speed = []

for file in tqdm(file_lists):
    # Create the file path
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)

    # Extract time, latitude, longitude, speed, acceleration, total distance, and power
    t, lat, log, v, a, total_distance, Power = data.T

    # Normalize the acceleration
    a_normalized = a / np.max(np.abs(a))

    # Save the average acceleration and average speed for each file
    all_average_acceleration.append(np.mean(a_normalized))
    all_average_speed.append(np.mean(v) * 3.6)  # Convert m/s to km/h

# Calculate the mean of average speed
average_speed_mean = np.mean(all_average_speed)

# Plotting the histogram of average acceleration
plt.figure()
sns.histplot(all_average_acceleration, bins='auto', color='red', kde=False)
plt.xlabel('Normalized Average Acceleration')
plt.ylabel('Number of trips')
plt.title('Normalized Average Acceleration Distribution')
plt.xlim(-0.02, 0.02)
plt.legend()
plt.grid(False)
plt.show()

# # Plotting the histogram of average speed
# plt.figure()
# sns.histplot(all_average_speed, bins='auto', color='purple', kde=True)
# plt.axvline(average_speed_mean, color='blue', linestyle='--', label='Avg Speed Mean')
# plt.text(average_speed_mean + 1, plt.gca().get_ylim()[1] * 0.8, f'{average_speed_mean:.2f}', color='blue')
# plt.xlabel('Average Speed (km/h)')
# plt.ylabel('Number of trips')
# plt.title('Average Speed Distribution')
# plt.xlim(0, 130)
# plt.legend()
# plt.grid(False)
# plt.show()