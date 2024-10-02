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
from matplotlib.patches import Rectangle  # 추가된 부분

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('GOOGLE_API_KEY')

# List of CSV files to process
csv_files = [
    r"D:\SamsungSTF\Data\Cycle\City_KOTI\20190101_240493.csv",
    r"D:\SamsungSTF\Data\Cycle\HW_KOTI\20190119_1903235.csv"
]

# Save path for the combined figure
save_path = r"C:\Users\BSL\Desktop\Figures\figure9.png"

# Labels for subplots
labels = ['A', 'B', 'C', 'D', 'E', 'F']

# Create a 2x3 Figure and Axes
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()  # Flatten to 1D array for easy indexing

for idx, csv_file in enumerate(csv_files):
    # Read CSV file
    df = pd.read_csv(csv_file)

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

    # Determine the starting index for subplots
    subplot_start = idx * 3

    ### 첫 번째 플롯: Google Maps에 경로 그리기 ###
    ax_map = axes[subplot_start]

    # Extract sampled latitudes and longitudes as lists
    latitudes = df_sampled['latitude'].tolist()
    longitudes = df_sampled['longitude'].tolist()

    if latitudes and longitudes:
        # Calculate center as the midpoint of the coordinates
        center_lat = np.mean(latitudes)
        center_lon = np.mean(longitudes)
        map_center = f"{center_lat},{center_lon}"

        # Set zoom level based on the row (idx)
        zoom = 14 if idx == 0 else 11  # 첫 번째 파일은 zoom=14, 두 번째 파일은 zoom=11

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
            ax_map.imshow(img)
            ax_map.axis('off')
            ax_map.set_title(f'Route Map {idx + 1}')

            # 검은색 테두리 추가
            rect = Rectangle((0, 0), 1, 1, transform=ax_map.transAxes, linewidth=2, edgecolor='black', facecolor='none')
            ax_map.add_patch(rect)
        else:
            ax_map.text(0.5, 0.5, 'Failed to load map', horizontalalignment='center', verticalalignment='center')
            ax_map.set_axis_off()
    else:
        ax_map.text(0.5, 0.5, 'No sampled coordinates', horizontalalignment='center', verticalalignment='center')
        ax_map.set_axis_off()

    # Add label
    ax_map.text(-0.1, 1.05, labels[subplot_start], transform=ax_map.transAxes, fontsize=16, fontweight='bold',
                va='bottom', ha='right')

    ### 두 번째 플롯: 속도와 가속도 (전체 데이터) ###
    ax_speed = axes[subplot_start + 1]
    ax_accel = ax_speed.twinx()

    # 속도: km/h로 변환 (m/s * 3.6)
    speed_kmh = df['speed'] * 3.6
    ax_speed.plot(df['elapsed_time_min'], speed_kmh, color='tab:blue', label='Speed (km/h)')

    # 가속도: tab:red
    ax_accel.plot(df['elapsed_time_min'], df['acceleration'], color='tab:red', label='Acceleration')

    ax_speed.set_xlabel('Elapsed Time (min)')
    ax_speed.set_ylabel('Speed (km/h)', color='tab:blue')
    ax_accel.set_ylabel('Acceleration', color='tab:red')

    ax_speed.tick_params(axis='y', labelcolor='tab:blue')
    ax_accel.tick_params(axis='y', labelcolor='tab:red')

    ax_speed.set_title(f'Speed and Acceleration')

    # Add legend
    lines_1, labels_1 = ax_speed.get_legend_handles_labels()
    lines_2, labels_2 = ax_accel.get_legend_handles_labels()
    ax_speed.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    # Add label
    ax_speed.text(-0.1, 1.05, labels[subplot_start + 1], transform=ax_speed.transAxes, fontsize=16, fontweight='bold',
                  va='bottom', ha='right')

    ### 세 번째 플롯: 파워 하이브리드 (전체 데이터) ###
    ax_power = axes[subplot_start + 2]

    # 파워 하이브리드: tab:green
    ax_power.plot(df['elapsed_time_min'], df['Power_hybrid']/1000, color='tab:green')
    ax_power.set_xlabel('Elapsed Time (min)')
    ax_power.set_ylabel('Power(kW)', color='tab:green')
    ax_power.set_title(f'Power Over Time')
    ax_power.tick_params(axis='y', labelcolor='tab:green')

    # Add label
    ax_power.text(-0.1, 1.05, labels[subplot_start + 2], transform=ax_power.transAxes, fontsize=16, fontweight='bold',
                  va='bottom', ha='right')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig(save_path, dpi=300)

plt.show()
