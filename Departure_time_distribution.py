import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Folder paths
win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터 Processed'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

departure_minutes = []

for file in file_lists:
    # Create the file path
    file_path = os.path.join(folder_path, file)
    T = np.loadtxt(file_path, delimiter=',', dtype=str, usecols=0)

    # Get the first T value (departure time)
    departure_time_str = T[0]

    # Convert to datetime object
    departure_time = datetime.strptime(departure_time_str, "%Y-%m-%d %H:%M:%S")

    # Store the hour:minute value of the departure time (integer value from 0 to 1439)
    departure_minutes.append(departure_time.hour * 60 + departure_time.minute)

# Histogram of departure time distribution
plt.figure()
sns.histplot(departure_minutes, bins=range(0, 1441, 60), color='purple', kde=False)
plt.xlabel('Departure Time (HH:MM)')
plt.ylabel('Number of Trips')
plt.title('Departure Time Distribution')
plt.xticks(range(0, 1440, 60), [f"{h:02d}:00" for h in range(0, 24)], rotation=45)
plt.grid(False)
plt.show()
