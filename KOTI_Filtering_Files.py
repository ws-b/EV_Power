import os
import shutil
import pandas as pd
import glob
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# Paths provided by the user
folder_path = r'D:\SamsungSTF\Processed_Data\KOTI'
city_save_path = r'D:\SamsungSTF\Processed_Data\BSL_Cycle\20km'
hw_save_path = r'D:\SamsungSTF\Processed_Data\BSL_Cycle\80km'

# Get list of all CSV files in folder_path
file_lists = glob.glob(os.path.join(folder_path, "*.csv"))

# Arrays to track files for each condition
city_files = []
hw_files = []
mean_speeds_within_time_range = []

# Process each file
for file in tqdm(file_lists):
    # Read the CSV file
    df = pd.read_csv(file)

    # Check if 'spd' and 'time' columns exist
    if 'spd' in df.columns and 'time' in df.columns:
        # Calculate the mean of the 'spd' column
        mean_spd = df['spd'].mean()

        try:
            # Convert the 'time' column to datetime
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

            # Calculate the time difference between the first and last row
            time_diff = df['time'].iloc[-1] - df['time'].iloc[0]
            time_diff_minutes = time_diff.total_seconds() / 60

            # Check if the time difference is between 10 to 20 minutes
            if 20 <= time_diff_minutes <= 40:
                # Classify based on the mean speed
                if mean_spd >= 50:
                    hw_files.append(file)
                    shutil.copy(file, hw_save_path)  # Copy to highway folder
        except Exception as e:
            # In case of any issues with the time column conversion or calculation
            continue
#
# for file in tqdm(file_lists):
#     # Read the CSV file
#     df = pd.read_csv(file)
#
#     # Check if 'spd' and 'time' columns exist
#     if 'spd' in df.columns and 'time' in df.columns:
#         try:
#             # Convert the 'time' column to datetime
#             df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
#
#             # Calculate the time difference between the first and last row
#             time_diff = df['time'].iloc[-1] - df['time'].iloc[0]
#             time_diff_minutes = time_diff.total_seconds() / 60
#
#             # Check if the time difference is between 20 to 35 minutes
#             if 20 <= time_diff_minutes <= 45:
#                 # Calculate the mean speed and store it
#                 mean_spd = df['spd'].mean()
#                 mean_speeds_within_time_range.append(mean_spd)
#         except Exception as e:
#             # In case of any issues with the time column conversion or calculation
#             continue
#
# # Plot the distribution of mean speeds
# plt.figure(figsize=(10, 6))
# plt.hist(mean_speeds_within_time_range, bins=30, color='blue', alpha=0.7)
# plt.title('Distribution of Mean Speeds (20-35 Minutes Duration)', fontsize=14)
# plt.xlabel('Mean Speed (km/h)', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.grid(True)
# plt.show()