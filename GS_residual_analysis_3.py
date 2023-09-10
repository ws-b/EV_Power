import os
import platform
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from GS_preprocessing import get_file_list
from tqdm import tqdm
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

# Use a random file for each key
selected_files = {key: random.choice(files) for key, files in grouped_files.items()}
dataframes = {}

# Process the data
for key, file in selected_files.items():
    df = pd.read_csv(os.path.join(folder_path, file))
    df['Residual'] = (df['Power_IV'] - df['Power']) / abs(df['Power_IV']).mean()
    dataframes[key] = df

# Plotting
for key, df in dataframes.items():
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    sns.kdeplot(df['Residual'], label=vehicle_types.get(key, 'unknown'), fill=True)

    plt.xlabel('Residual')
    plt.ylabel('Density')
    plt.title(f'Density Plot of Residuals for {vehicle_types.get(key, "unknown")}')
    plt.xlim(-6, 6)
    plt.legend()
    plt.grid(True)
    plt.show()

print("Done")
