import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
import matplotlib.font_manager as fm
import numpy as np

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('GOOGLE_API_KEY')

# Read CSV file
csv_file = r"D:\SamsungSTF\Data\Cycle\City_KOTI\20190101_240493.csv"
df = pd.read_csv(csv_file)
#save_path
save_path = r"C:\Users\BSL\Desktop\figure9.png"

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

# Create Figure and Axes
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

### 첫 번째 플롯: Google Maps에 경로 그리기 ###
# Extract sampled latitudes and longitudes as lists
latitudes = df_sampled['latitude'].tolist()
longitudes = df_sampled['longitude'].tolist()

if latitudes and longitudes:
    # Calculate center as the midpoint of the coordinates
    center_lat = np.mean(latitudes)
    center_lon = np.mean(longitudes)
    map_center = f"{center_lat},{center_lon}"

    # Increase zoom level for closer view
    zoom = 14

    # Increase size for higher resolution if needed
    size = "800x800"

    # Convert coordinates to 'lat,lon' format separated by '|'
    path = '|'.join([f"{lat},{lon}" for lat, lon in zip(latitudes, longitudes)])

    # Static Maps API request URL with red color and thicker path
    map_url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={map_center}&zoom={zoom}&size={size}"
        f"&path=color:0xff0000|weight:4|{path}"  # 빨간색(0xff0000)과 두께(weight:4)
        f"&language=en"  # 지명 레이블을 영어로 설정
        f"&key={API_KEY}"
    )

    # Fetch map image
    response = requests.get(map_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title('Route Map')
    else:
        axes[0].text(0.5, 0.5, 'Failed to load map', horizontalalignment='center', verticalalignment='center')
        axes[0].set_axis_off()
else:
    axes[0].text(0.5, 0.5, 'No sampled coordinates', horizontalalignment='center', verticalalignment='center')
    axes[0].set_axis_off()

# Add 'A' label slightly outside the plot area
axes[0].text(-0.02, 1.05, 'A', transform=axes[0].transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')

### 두 번째 플롯: 속도와 가속도 (전체 데이터) ###
ax1 = axes[1]
ax2 = ax1.twinx()

# 속도: km/h로 변환 (m/s * 3.6)
speed_kmh = df['speed'] * 3.6
ax1.plot(df['elapsed_time_min'], speed_kmh, color='tab:blue', label='Speed (km/h)')  # 'tab:blue'

# 가속도: tab:red
ax2.plot(df['elapsed_time_min'], df['acceleration'], color='tab:red', label='Acceleration')  # 'tab:red'

ax1.set_xlabel('Elapsed Time (min)')
ax1.set_ylabel('Speed (km/h)', color='tab:blue')  # 'tab:blue'
ax2.set_ylabel('Acceleration', color='tab:red')  # 'tab:red'

ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:red')

axes[1].set_title('Speed and Acceleration')

# Add legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

# Add 'B' label slightly outside the plot area
axes[1].text(-0.02, 1.05, 'B', transform=axes[1].transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')

### 세 번째 플롯: 파워 하이브리드 (전체 데이터) ###
# 파워 하이브리드: tab:green
axes[2].plot(df['elapsed_time_min'], df['Power_hybrid']/1000, color='tab:green')
axes[2].set_xlabel('Elapsed Time (min)')
axes[2].set_ylabel('Power(kW)', color='tab:green')
axes[2].set_title('Power Over Time')
axes[2].tick_params(axis='y', labelcolor='tab:green')

# Add 'C' label slightly outside the plot area
axes[2].text(-0.02, 1.05, 'C', transform=axes[2].transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')

# Adjust layout to accommodate labels outside the plot area
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
