import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import contextily as ctx
from shapely.geometry import Point

# Define the path to the uploaded file
file_path = r"C:\Users\WSONG\Desktop\Cycle\HW_KOTI\20190119_1903235.csv"

# Load the CSV file
df = pd.read_csv(file_path)

# Create a GeoDataFrame with the latitude and longitude
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")

# Convert to Web Mercator (the coordinate system used by most web maps, including OpenStreetMap)
gdf = gdf.to_crs(epsg=3857)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 4))

# Plot the points from the CSV file
gdf.plot(ax=ax, color='red', markersize=5, label='Route')

# Add English OpenStreetMap basemap (using Stamen or CartoDB)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)  # You can also try ctx.providers.Stamen.Terrain

# Hide x and y axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Set the title and labels
plt.title('GPS route')
plt.tight_layout()
plt.show()
