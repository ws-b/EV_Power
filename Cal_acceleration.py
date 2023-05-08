import os
import numpy as np

mac_folder_path = '/Users/woojin/Desktop/대학교 자료/켄텍 자료/삼성미래과제/Driving Pattern/Drive Cycle/'
mac_save_path = '/Users/woojin/Desktop/대학교 자료/켄텍 자료/삼성미래과제/Driving Pattern/Drive Cycle Processed/'
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 Processed\\'
win_save_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'

folder_path = mac_folder_path
save_path = mac_save_path
# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]
for file_list in file_lists:
    file = open(folder_path + file_list, "r")

    # Load CSV file into a numpy array
    data = np.genfromtxt(file, delimiter='\t')

    # Extract columns into separate arrays
    times = data[:, 0]
    speeds = data[:, 1]
    # Calculate acceleration using finite difference method
    acceleration = np.diff(speeds)

    # To save the calculated acceleration data, you can append it as a new column to the existing data array.
    # First, create a new array with the same length as the original data array.
    extended_acceleration = np.zeros_like(times)

    # Fill the new array with acceleration values, leaving the last element as zero.
    extended_acceleration[:-1] = acceleration

    # Combine the original data array with the new acceleration column.
    combined_data = np.column_stack((data, extended_acceleration))

    # Save the combined data to a new CSV file in the save_path directory.
    output_file_path = os.path.join(save_path, f"acceleration_{file_list[:-4]}.csv")
    np.savetxt(output_file_path, combined_data, delimiter=',', fmt='%.4f')