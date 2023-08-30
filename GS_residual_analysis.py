import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from GS_preprocessing import get_file_list
from tqdm import tqdm
from collections import defaultdict

def select_vehicle(num):
    vehicles = {1: 'ioniq 5', 2: 'kona EV', 3: 'porter EV'}
    return vehicles.get(num, 'unknown')

folder_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\trip_by_trip'
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

save_folder = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터'
merged_dataframes = {}

for key, files in tqdm(grouped_files.items()):
    list_of_dfs = [pd.read_csv(os.path.join(folder_path, f)) for f in files]
    merged_df = pd.concat(list_of_dfs, ignore_index=True)
    merged_dataframes[key] = merged_df
    merged_df['Residual'] = merged_df['Power'] - merged_df['Power_IV']
    vehicle_type = vehicle_types.get(key, 'unknown')

    # Save merged dataframe to CSV
    merged_df.sort_values(by='time', ignore_index=True)
    merged_df.to_csv(os.path.join(save_folder, f"{key}.csv"), index=False)

    plt.figure(figsize=(10, 6))
    sns.regplot(data=merged_df, x='speed', y='Residual', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title('Relationship between Speed and Residuals')
    plt.xlabel('Speed (mps)')
    plt.ylabel('Residual (Power - Power_IV)')
    plt.text(0.95, 0.95, f'{vehicle_type}', ha='right', va='top', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.regplot(data=merged_df, x='acceleration', y='Residual', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title('Relationship between Acceleration and Residuals')
    plt.xlabel('Acceleration (mps^2)')
    plt.ylabel('Residual (Power - Power_IV)')
    plt.text(0.95, 0.95, f'{vehicle_type}', ha='right', va='top', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.regplot(data=merged_df, x='ext_temp', y='Residual', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title('Relationship between External Temperature and Residuals')
    plt.xlabel('External Temperature (°C)')
    plt.ylabel('Residual (Power - Power_IV)')
    plt.text(0.95, 0.95, f'{vehicle_type}', ha='right', va='top', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.show()

print("Done")