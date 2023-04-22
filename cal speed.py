import os
import numpy as np
from geopy import Point
from geopy import distance
from pyproj import Transformer

mac_folder_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터 Processed/'
mac_save_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터 속도:가속도 처리/'
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 Processed\\'
win_save_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'
folder_path = mac_folder_path
save_path = mac_save_path
# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
#file_list = '164_2.csv'
for file_list in file_lists:
    file = open(folder_path + file_list, "r")

    # Load CSV file into a numpy array
    data = np.genfromtxt(file, delimiter=',')

    # Extract columns into separate arrays
    times = data[:, 0]
    latitudes = data[:, 1]
    longitudes = data[:, 2]

    # convert the latitude and longitude data to UTM coordinates
    lat = []
    log = []
    for i in range (0, len(latitudes)):
        transformer = Transformer.from_crs("epsg:5179", "epsg:4326")
        utm_lat, utm_log = transformer.transform(latitudes[i], longitudes[i])
        # UTM-K(Bassel) 도로명주소 지도 사용 중 to Wgs84 경도/위도, GPS사용 전지구 좌표
        lat.append(utm_lat)
        log.append(utm_log)

    # Calculate distance between each pair of consecutive UTM coordinates
    speeds = [distance.distance(Point(lat[i], log[i]), Point(lat[i+1], log[i+1])).m / (times[i+1]-times[i]) for i in range(0,len(lat)-1)]
    speeds.append(speeds[-1])

    # Calculate the acceleration by taking the derivative of the speeds with respect to time
    accelerations = [((speeds[i+1] - speeds[i]) / (times[i+1] - times[i])) for i in range(len(speeds)-1)]
    accelerations.append(accelerations[-1])

    # Use numpy.column_stack() to append the list as a new column
    data = np.column_stack((data, speeds))
    data = np.column_stack((data, accelerations))

    # Export the array to a text file
    np.savetxt(save_path + file_list, data, delimiter=",", fmt='%.8f')