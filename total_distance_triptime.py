import os
import numpy as np

# Folder path containing the files
win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터 속도-가속도 처리자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'
mac_folder_path = ''
folder_path = os.path.normpath(mac_folder_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# Iterate over the files
for file in file_lists:
    # Create the file path
    file_path = os.path.join(folder_path, file)

    # Open the file
    with open(file_path, 'r') as f:
        # Read the file
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)
        total_distance = np.cumsum(data[:, 1])
        data = np.column_stack((data, total_distance))
        np.savetxt(file_path, data, delimiter=',', fmt='%f')
