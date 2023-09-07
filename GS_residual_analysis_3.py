import os
import platform
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from GS_preprocessing import get_file_list
from tqdm import tqdm
from collections import defaultdict

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

for file in file_lists:

    merged_df['Residual'] = (merged_df['Power_IV'] - merged_df['Power']) / abs(merged_df['Power_IV']).mean()
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    sns.kdeplot(df['Residual'], fill=True)

    plt.xlabel('Residual')
    plt.ylabel('Density')
    plt.title(f'Density Plot of Residuals for {vehicle_types.get(key, "unknown")}')
    plt.legend()
    plt.grid(True)
    plt.show()
