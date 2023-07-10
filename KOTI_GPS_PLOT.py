import os
import pandas as pd
import folium
from pyproj import Transformer
from tqdm import tqdm

win_folder_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터_후처리\230710'
folder_path = os.path.normpath(win_folder_path)

win_save_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터_후처리\230710\map'
save_path = os.path.normpath(win_save_path)

# check if save_path exists
if not os.path.exists(save_path):
    # if not, create the directory
    os.makedirs(save_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# Open each CSV file
for file in tqdm(file_lists[10:20]):
    file_path = os.path.join(folder_path, file)

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path, header=None)

    # Set column names 'time', 'longitude', 'latitude', etc.
    data.columns = ['time', 'longitude', 'latitude', 'speed', 'acceleration', 'type', 'vLinkId', 'distFromLink',
                    'distToLink', 'gisLength', 'gisSpeed', 'gpsLength', 'gpsSpeed', 'Energy']

    # Transform coordinates from epsg:5179 (UTM coordinates) to epsg:4326 (WGS 84 coordinates)
    transformer = Transformer.from_crs("epsg:5179", "epsg:4326")
    data['latitude'], data['longitude'] = transformer.transform(data['latitude'].values, data['longitude'].values)

    # Define a valid range of latitude and longitude for South Korea
    valid_latitude_range = [32, 43]  # replace with your valid latitude range
    valid_longitude_range = [123, 134]  # replace with your valid longitude range

    # # Get the indices of data points outside the valid range
    # invalid_indices = data.index[~((data['latitude'] >= valid_latitude_range[0]) &
    #                                (data['latitude'] <= valid_latitude_range[1]) &
    #                                (data['longitude'] >= valid_longitude_range[0]) &
    #                                (data['longitude'] <= valid_longitude_range[1]))].tolist()
    #
    # print(f"Invalid indices in {file}: {invalid_indices}")

    # # Remove data points outside the valid range
    data = data[(data['latitude'] >= valid_latitude_range[0]) &
                (data['latitude'] <= valid_latitude_range[1]) &
                (data['longitude'] >= valid_longitude_range[0]) &
                (data['longitude'] <= valid_longitude_range[1])]

    # Extract latitude and longitude information from the DataFrame
    locations = data[['latitude', 'longitude']].values.tolist()

    # Create a map with the first location as the center (latitude, longitude order)
    m = folium.Map(location=[locations[0][0], locations[0][1]], zoom_start=13)

    # Add a line connecting each location with a thicker line
    folium.PolyLine(locations, color="blue", weight=5, opacity=1).add_to(m)  # Increase weight to 5 for a thicker line

    # Save the map as an HTML file
    m.save(os.path.join(save_path, f"{file}.html"))
