import os
import numpy as np
from geopy import Point
from geopy import distance
from pyproj import Transformer

win_folder_path = 'G:\\공유 드라이브\\Battery Software Lab\\Data\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 Processed\\'
win_save_path = 'G:\\공유 드라이브\\Battery Software Lab\\Data\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'
folder_path = os.path.normpath(win_folder_path)
save_path = os.path.normpath(win_save_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

for file in file_lists:
    file_path = os.path.join(folder_path, file)

    # Load CSV file into a numpy array
    data = np.genfromtxt(file_path, delimiter=',')

    # Extract columns into separate arrays
    times = data[:, 0]
    latitudes = data[:, 1]
    longitudes = data[:, 2]

    # Convert UTM coordinates to WGS84 coordinates
    wgs84_latitudes = []
    wgs84_longitudes = []
    transformer = Transformer.from_crs("epsg:5179", "epsg:4326")
    for i in range(len(latitudes)):
        wgs84_lat, wgs84_long = transformer.transform(latitudes[i], longitudes[i])
        wgs84_latitudes.append(wgs84_lat)
        wgs84_longitudes.append(wgs84_long)

    # Calculate distance between each pair of consecutive WGS84 coordinates
    speeds = [distance.distance(Point(wgs84_latitudes[i], wgs84_longitudes[i]), Point(wgs84_latitudes[i + 1], wgs84_longitudes[i + 1])).m / (times[i + 1] - times[i]) for i in range(0, len(wgs84_latitudes) - 1)]
    speeds.append(speeds[-1])

    # Calculate the acceleration by taking the derivative of the speeds with respect to time
    accelerations = [((speeds[i + 1] - speeds[i]) / (times[i + 1] - times[i])) for i in range(len(speeds) - 1)]
    accelerations.append(accelerations[-1])

    # Use numpy.column_stack() to append the list as a new column
    data = np.column_stack((data, speeds))
    data = np.column_stack((data, accelerations))

    # Export the array to a text file
    file_save_path = os.path.join(save_path, file)
    np.savetxt(file_save_path, data, delimiter=",", fmt='%.8f')