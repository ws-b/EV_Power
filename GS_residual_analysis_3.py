import os
import platform
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from GS_preprocessing import get_file_list
from collections import defaultdict
import random

def select_vehicle(num):
    vehicles = {1: 'ioniq 5', 2: 'kona EV', 3: 'porter EV'}
    return vehicles.get(num, 'unknown')

if platform.system() == "Windows":
    folder_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\trip_by_trip')
elif platform.system() == "Darwin":
    folder_path = os.path.normpath(
        '/Users/wsong/Documents/KENTECH/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip')
else:
    print("Unknown system.")

file_lists = get_file_list(folder_path)

vehicle_types = {
    '01241248726': 'kona EV',
    '01241248782': 'ioniq 5',
    '01241228177': 'porter EV'
}

file_lists = [file for file in file_lists if any(key in file for key in vehicle_types.keys())]
grouped_files = defaultdict(list)

for file in file_lists:
    key = file[:11]
    grouped_files[key].append(file)

# Get merged dataframes for all data
merged_dataframes = {}
for key, files in grouped_files.items():
    list_of_dfs = [pd.read_csv(os.path.join(folder_path, f)) for f in files]
    merged_df = pd.concat(list_of_dfs, ignore_index=True)
    merged_df['Residual'] = (merged_df['Power_IV'] - merged_df['Power']) / abs(merged_df['Power_IV']).mean()
    merged_dataframes[key] = merged_df

# Select random files
selected_files = {key: random.choices(files, k=5) for key, files in grouped_files.items()}

# Process the data and plot for each vehicle type
for key in vehicle_types.keys():
    if key not in merged_dataframes:  # Check if the key exists in merged_dataframes
        continue

    plt.figure(figsize=(10, 7))
    sns.kdeplot(merged_dataframes[key]['Residual'], label=f'All {vehicle_types.get(key, "unknown")}', fill=True)

    # Process the data for random files
    for idx, file in enumerate(selected_files[key]):
        df = pd.read_csv(os.path.join(folder_path, file))
        df['Residual'] = (df['Power_IV'] - df['Power']) / abs(df['Power_IV']).mean()
        sns.kdeplot(df['Residual'], label=f'Sample {idx+1}')

    plt.xlabel('Residual')
    plt.ylabel('Density')
    plt.title(f'Density Plot of Residuals for {vehicle_types.get(key, "unknown")}')
    plt.xlim(-6, 6)
    plt.legend()
    plt.grid(True)
    plt.show()

print("Done")