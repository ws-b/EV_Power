import os
import pandas as pd
from pyproj import Transformer

win_folder_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터_후처리\230706'
folder_path = os.path.normpath(win_folder_path)

log_path = os.path.join(win_folder_path, 'log')

if not os.path.exists(log_path):
    os.makedirs(log_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# Open each CSV file
for file in file_lists:
    file_path = os.path.join(folder_path, file)

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path, header=None)

    # Set column names 'time', 'longitude', 'latitude', etc.
    data.columns = ['time', 'longitude', 'latitude', 'speed', 'acceleration', 'type', 'vLinkId', 'distFromLink',
                    'distToLink', 'gisLength', 'gisSpeed', 'gpsLength', 'gpsSpeed']

    # Transform coordinates from epsg:5179 (UTM coordinates) to epsg:4326 (WGS 84 coordinates)
    transformer = Transformer.from_crs("epsg:5179", "epsg:4326")
    data['latitude'], data['longitude'] = transformer.transform(data['latitude'].values, data['longitude'].values)

    # Define a valid range of latitude and longitude for South Korea
    valid_latitude_range = [32, 43]  # replace with your valid latitude range
    valid_longitude_range = [123, 134]  # replace with your valid longitude range

    # Create boolean mask for data points outside the valid range
    invalid_mask = ~((data['latitude'] >= valid_latitude_range[0]) &
                     (data['latitude'] <= valid_latitude_range[1]) &
                     (data['longitude'] >= valid_longitude_range[0]) &
                     (data['longitude'] <= valid_longitude_range[1]))

    print(f"Invalid indices in {file}: {invalid_mask[invalid_mask].index.tolist()}")

    # If there are any invalid data points, save them in a log file
    if invalid_mask.any():
        # Save rows that will be removed into a separate log file
        removed_data = data[invalid_mask]
        removed_data.to_csv(os.path.join(log_path, f"removed_{file}"), index=False)

    # Keep only the valid data points
    valid_data = data[~invalid_mask]

    # Overwrite the original CSV file with the valid data
    valid_data.to_csv(file_path, index=False)
