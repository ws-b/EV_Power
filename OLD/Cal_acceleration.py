import os
import numpy as np

win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터 Processed'
win_save_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터 속도-가속도 처리'

folder_path = os.path.normpath(win_folder_path)
save_path = os.path.normpath(win_save_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

for file in file_lists:
    file_path = os.path.join(folder_path, file)

    # Load CSV file into a numpy array
    data = np.genfromtxt(file_path, delimiter='\t')

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
    output_file_path = os.path.join(save_path, f"acceleration_{file[:-4]}.csv")
    np.savetxt(output_file_path, combined_data, delimiter=',', fmt='%.4f')