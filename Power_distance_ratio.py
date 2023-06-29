import os
import numpy as np

win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터 속도-가속도 처리'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

distance_per_total_power = []

for file in file_lists:
    # Create the file path
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)

    # Extract time, speed, acceleration, total distance, and power
    t, v, a, total_distance, Power = data.T

    # Calculate the total power
    total_power = np.sum(Power)

    ratio = data[-1, 3] / total_power
    print("Ratio: ", ratio)
