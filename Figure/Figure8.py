import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import folium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import io
from PIL import Image

# Read CSV file
csv_file = r"D:\SamsungSTF\Data\Cycle\City_KOTI\20190101_240493.csv"
df = pd.read_csv(csv_file)

# Save path
save_path = r"C:\Users\BSL\Desktop\Figures\figure8_temp.png"

# Convert 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

# Calculate elapsed time based on start time (in seconds)
start_time = df['time'].iloc[0]
df['elapsed_time_sec'] = (df['time'] - start_time).dt.total_seconds()

# Convert elapsed time to minutes
df['elapsed_time_min'] = df['elapsed_time_sec'] / 60

# Set the dataframe index to time
df.set_index('time', inplace=True)

# Sample coordinate data at 10-second intervals
df_sampled = df.resample('10s').first().dropna(subset=['latitude', 'longitude'])

# Create Figure and Axes with adjusted figsize
fig, axes = plt.subplots(3, 1, figsize=(15, 15))

# Adjust the vertical space between subplots
plt.subplots_adjust(hspace=0.4)

### First Plot: Plotting Route with Folium ###
latitudes = df_sampled['latitude'].tolist()
longitudes = df_sampled['longitude'].tolist()

if latitudes and longitudes:
    # Calculate center for the map
    center_lat = np.mean(latitudes)
    center_lon = np.mean(longitudes)

    # Create a Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)

    # Add the route as a PolyLine
    route = list(zip(latitudes, longitudes))
    folium.PolyLine(route, color="red", weight=4).add_to(m)

    # Save map to HTML and then render to PNG
    m.save('map.html')

    # Set up Selenium WebDriver
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    # Load the map HTML file
    driver.get('file://' + os.path.realpath('map.html'))

    # Set the window size to match the figure width
    map_width = 1200  # 15.9 inches * 100 dpi
    map_height = 400  # Adjust height as needed
    driver.set_window_size(map_width, map_height)

    # Give the map some time to load
    driver.implicitly_wait(5)

    # Take a screenshot
    png = driver.get_screenshot_as_png()
    driver.quit()

    # Read the image
    img = Image.open(io.BytesIO(png))

    # Crop the image to remove browser UI elements if necessary
    img = img.crop((0, 0, map_width, map_height))

    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('Route Map')
else:
    axes[0].text(0.5, 0.5, 'No sampled coordinates', horizontalalignment='center', verticalalignment='center')
    axes[0].set_axis_off()

### Second Plot: Speed and Acceleration ###
ax1 = axes[1]
ax2 = ax1.twinx()

# Speed in km/h
speed_kmh = df['speed'] * 3.6
ax1.plot(df['elapsed_time_min'], speed_kmh, color='tab:blue', label='Speed (km/h)')

# Acceleration
ax2.plot(df['elapsed_time_min'], df['acceleration'], color='tab:red', label='Acceleration')

ax1.set_xlabel('Elapsed Time (min)')
ax1.set_ylabel('Speed (km/h)', color='tab:blue')
ax2.set_ylabel('Acceleration', color='tab:red')

ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:red')

axes[1].set_title('Speed and Acceleration')

# Add legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')


### Third Plot: Power Hybrid ###
axes[2].plot(df['elapsed_time_min'], df['Power_hybrid']/1000, color='tab:green')
axes[2].set_xlabel('Elapsed Time (min)')
axes[2].set_ylabel('Power (kW)', color='tab:green')
axes[2].set_title('Power Over Time')
axes[2].tick_params(axis='y', labelcolor='tab:green')


# Adjust layout
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
