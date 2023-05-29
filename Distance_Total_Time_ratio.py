import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

all_total_distance = []
all_total_time = []

for file in files:
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
