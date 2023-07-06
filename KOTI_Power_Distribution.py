import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

win_folder_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터_후처리\230706'

folder_path = os.path.normpath(win_folder_path)

# Get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_distance_per_total_Energy = []
over_50_files = []

for file in file_lists:
    # Create the file path
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # Convert 'time' column to datetime
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

    # Extract time, latitude, longitude, speed, acceleration, total distance, and Energy
    speed = data['speed']  # assuming the speed column is named 'speed'
    Energy = data['Energy']

    # Calculate time difference in seconds
    time_difference = data['time'].diff().dt.total_seconds().iloc[1:]  # discard the first NaN value

    # Calculate total distance by summing (speed * time_difference) for each step
    distance_per_step = speed.iloc[:-1] * time_difference  # calculate distance for each step
    total_distance = distance_per_step.sum()  # calculate total distance

    # Calculate the total Energy
    total_Energy = Energy.sum()

    # Calculate the total time as difference between max and min time
    total_time = (data['time'].max() - data['time'].min()).total_seconds()

    # Calculate Total distance / Total Energy for each file (if total_Energy is 0, set the value to 0)
    distance_per_total_Energy_km_kWh = (total_distance / 1000) / (total_Energy) if total_Energy != 0 else 0

    # If distance_per_total_Energy_km_kWh is greater than or equal to 50, add the file name to over_50_files
    if distance_per_total_Energy_km_kWh >= 50:
        over_50_files.append(file)
    # Collect all distance_per_total_Energy values
    all_distance_per_total_Energy.append(distance_per_total_Energy_km_kWh)

# Create a histogram for all files
sns.histplot(all_distance_per_total_Energy, bins='auto', color='gray', kde=False)

# Draw a vertical line for the mean
mean_value = pd.Series(all_distance_per_total_Energy).mean()
plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')

# Display the mean value
plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red', fontsize=12)

plt.xlabel('Total Distance / Total Energy (km/kWh)')
plt.ylabel('Number of trips')
plt.title('Total Distance / Total Energy Distribution')
plt.grid(False)
plt.show()

# Print files with a ratio of Total Distance / Total Energy greater than 50
print("Files with a ratio of Total Distance / Total Energy greater than 50:")
for over_50_file in over_50_files:
    print(over_50_file)
