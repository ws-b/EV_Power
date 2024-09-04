import os
import glob
import pandas as pd
import folium
from pyproj import Transformer
from tqdm import tqdm

folder_path = r'D:\SamsungSTF\Processed_Data\BSL_Cycle\20km'

folder_path = os.path.normpath(folder_path)
save_path = os.path.normpath(os.path.join(folder_path), 'map')

# check if save_path exists
if not os.path.exists(save_path):
    # if not, create the directory
    os.makedirs(save_path)

file_lists = glob.glob(os.path.join(folder_path, "*.csv"))

# Open each CSV file
for file in tqdm(file_lists):
    file_path = os.path.join(folder_path, file)

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path, header=None)

    # Transform coordinates from epsg:5179 (UTM coordinates) to epsg:4326 (WGS 84 coordinates)
    transformer = Transformer.from_crs("epsg:5179", "epsg:4326")
    data['y'], data['x'] = transformer.transform(data['y'].values, data['x'].values)

    # Define a valid range of latitude and longitude for South Korea
    valid_latitude_range = [32, 43]  # replace with your valid latitude range
    valid_longitude_range = [123, 134]  # replace with your valid longitude range

    # Remove data points outside the valid range
    data = data[(data['x'] >= valid_latitude_range[0]) &
                (data['x'] <= valid_latitude_range[1]) &
                (data['y'] >= valid_longitude_range[0]) &
                (data['y'] <= valid_longitude_range[1])]

    # Extract latitude and longitude information from the DataFrame
    locations = data[['y', 'x']].values.tolist()

    # Create a map with the first location as the center (latitude, longitude order)
    m = folium.Map(location=[locations[0][0], locations[0][1]], zoom_start=13)

    # Add a line connecting each location with a thicker line
    folium.PolyLine(locations, color="blue", weight=5, opacity=1).add_to(m)  # Increase weight to 5 for a thicker line

    # Save the map as an HTML file
    m.save(os.path.join(save_path, f"{file}.html"))
