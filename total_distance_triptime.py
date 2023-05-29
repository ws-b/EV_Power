import os
import numpy as np

# Folder path containing the files
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'
mac_folder_path = '/Users/woojin/Desktop/대학교 자료/켄텍 자료/삼성미래과제/Driving Pattern/Drive Cycle Processed/'
folder_path = mac_folder_path

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

# Iterate over the files
for file in files:
    # Create the file path
    file_path = os.path.join(folder_path, file)

    # Open the file
    with open(file_path, 'r') as f:
        # Read the file
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)
        total_distance = np.cumsum(data[:, 1])
        data = np.column_stack((data, total_distance))
        np.savetxt(file_path, data, delimiter=',', fmt='%f')
