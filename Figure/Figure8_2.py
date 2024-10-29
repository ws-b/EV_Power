import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
from PIL import Image
from io import BytesIO
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.patches import Rectangle

# List of CSV files to process
csv_files = [
    r"D:\SamsungSTF\Data\Cycle\City_KOTI\20190101_240493.csv",
    r"D:\SamsungSTF\Data\Cycle\HW_KOTI\20190119_1903235.csv"
]

# Save path for the combined figure
save_path = r"C:\Users\BSL\Desktop\Figures\figure8.png"

# Labels for subplots (updated for 2x4 grid)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# Create a 2x4 Figure and Axes with appropriate projections
fig = plt.figure(figsize=(24, 12))

# Define projection for map subplots
projection = ccrs.PlateCarree()

# Create axes manually to assign projections only to map subplots
axes = []
for i in range(8):
    if i % 4 == 0:
        ax = fig.add_subplot(2, 4, i+1, projection=projection)
    else:
        ax = fig.add_subplot(2, 4, i+1)
    axes.append(ax)

# Initialize OSM Tiles
osm_tiles = cimgt.OSM()

def add_north_arrow(ax, size=0.1, loc=(0.95, 0.95)):
    """
    Adds a north arrow to a Cartopy GeoAxes.

    Parameters:
    - ax: The GeoAxes to add the arrow to.
    - size: Size of the arrow.
    - loc: Location of the arrow in axis coordinates (x, y).
    """
    ax.annotate('N',
                xy=loc, xycoords='axes fraction',
                xytext=(loc[0], loc[1] - size),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.1'))

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

    # Determine the starting index for subplots (updated for 4 columns)
    subplot_start = idx * 4

    ### 첫 번째 플롯: Cartopy와 OSM을 사용하여 경로 그리기 ###
    ax_map = axes[subplot_start]

    # Extract sampled latitudes and longitudes as lists
    latitudes = df_sampled['latitude'].tolist()
    longitudes = df_sampled['longitude'].tolist()

    if latitudes and longitudes:
        # Calculate center as the midpoint of the coordinates
        center_lat = np.mean(latitudes)
        center_lon = np.mean(longitudes)

        # Set extent to include all points with some padding
        buffer = 0.01  # degrees
        min_lon, max_lon = min(longitudes) - buffer, max(longitudes) + buffer
        min_lat, max_lat = min(latitudes) - buffer, max(latitudes) + buffer
        ax_map.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        # Add OSM tiles as the background
        ax_map.add_image(osm_tiles, 12)  # 12은 확대 수준 (Zoom level)

        # Plot the route
        ax_map.plot(longitudes, latitudes, color='red', linewidth=2, marker='o', transform=ccrs.PlateCarree())

        # Add features
        ax_map.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
        ax_map.add_feature(cfeature.COASTLINE, edgecolor='gray')

        if idx == 0:
            ax_map.set_title('City Cycle Route Map')
        else:
            ax_map.set_title('Highway Cycle Route Map')

        # 검은색 테두리 추가
        rect = Rectangle((0, 0), 1, 1, transform=ax_map.transAxes, linewidth=2, edgecolor='black', facecolor='none')
        ax_map.add_patch(rect)

        # 방향 표시 추가
        add_north_arrow(ax_map, size=0.05, loc=(0.95, 0.95))
    else:
        ax_map.text(0.5, 0.5, 'No sampled coordinates', horizontalalignment='center', verticalalignment='center')
        ax_map.set_axis_off()

    # Add label
    if latitudes and longitudes:
        ax_map.text(-0.1, 1.05, labels[subplot_start], transform=ax_map.transAxes, fontsize=16, fontweight='bold',
                    va='bottom', ha='right')

    ### 두 번째 플롯: 속도와 가속도 (전체 데이터) ###
    ax_speed = axes[subplot_start + 1]
    ax_accel = ax_speed.twinx()

    # 속도: km/h로 변환 (m/s * 3.6)
    speed_kmh = df['speed'] * 3.6
    ax_speed.plot(df['elapsed_time_min'], speed_kmh, color='tab:blue', label='Speed')

    # 가속도: tab:red
    ax_accel.plot(df['elapsed_time_min'], df['acceleration'], color='tab:red', label='Acceleration')

    ax_speed.set_xlabel('Elapsed Time (min)')
    ax_speed.set_ylabel('Speed (km/h)', color='tab:blue')
    ax_accel.set_ylabel('Acceleration', color='tab:red')

    ax_speed.tick_params(axis='y', labelcolor='tab:blue')
    ax_accel.tick_params(axis='y', labelcolor='tab:red')

    ax_speed.set_title('Speed and Acceleration')

    # Add legend
    lines_1, labels_1 = ax_speed.get_legend_handles_labels()
    lines_2, labels_2 = ax_accel.get_legend_handles_labels()
    ax_speed.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    # Add label
    ax_speed.text(-0.1, 1.05, labels[subplot_start + 1], transform=ax_speed.transAxes, fontsize=16, fontweight='bold',
                  va='bottom', ha='right')

    ### 세 번째 플롯: 파워 하이브리드 및 파워 피직스 (전체 데이터) ###
    ax_power = axes[subplot_start + 2]

    # Plot Power_hybrid and Power_phys
    ax_power.plot(df['elapsed_time_min'], df['Power_phys']/1000, color='tab:red', label='Power Phys', alpha=0.7)
    ax_power.plot(df['elapsed_time_min'], df['Power_hybrid'] / 1000, color='tab:green', label='Power Hybrid', alpha=0.7)

    # Set labels and title
    ax_power.set_xlabel('Elapsed Time (min)')
    ax_power.set_ylabel('Power (kW)')
    ax_power.set_title('Power Over Time')

    # Add legend to distinguish between the two power metrics
    ax_power.legend(loc='upper right')

    # Optionally, set y-axis tick colors to default or customize as needed
    ax_power.tick_params(axis='y', labelcolor='black')

    # Add subplot label
    ax_power.text(-0.1, 1.05, labels[subplot_start + 2], transform=ax_power.transAxes, fontsize=16, fontweight='bold',
                  va='bottom', ha='right')

    ### 네 번째 플롯: 누적 에너지 (적분) ###
    ax_energy = axes[subplot_start + 3]

    # Compute cumulative energy using manual cumulative trapezoidal integration
    # Convert elapsed_time_min to hours for kWh calculation
    elapsed_time_hours = df['elapsed_time_min'] / 60

    # Initialize energy arrays
    energy_phys = [0]
    energy_hybrid = [0]

    # Iterate through the data to compute cumulative energy
    for i in range(1, len(df)):
        # Current and previous time points
        t_prev = elapsed_time_hours.iloc[i-1]
        t_curr = elapsed_time_hours.iloc[i]
        dt = t_curr - t_prev

        # Average power between current and previous points
        p_phys_avg = (df['Power_phys'].iloc[i-1] + df['Power_phys'].iloc[i]) / 2 / 1000  # kW
        p_hybrid_avg = (df['Power_hybrid'].iloc[i-1] + df['Power_hybrid'].iloc[i]) / 2 / 1000  # kW

        # Incremental energy
        energy_phys.append(energy_phys[-1] + p_phys_avg * dt)
        energy_hybrid.append(energy_hybrid[-1] + p_hybrid_avg * dt)

    # Convert to NumPy arrays for plotting
    energy_phys = np.array(energy_phys)
    energy_hybrid = np.array(energy_hybrid)

    # Align energy arrays with elapsed_time_min
    energy_phys = energy_phys[:len(df)]
    energy_hybrid = energy_hybrid[:len(df)]

    # Plot cumulative energy
    ax_energy.plot(df['elapsed_time_min'], energy_phys, color='tab:red', label='Energy Phys', alpha=0.7)
    ax_energy.plot(df['elapsed_time_min'], energy_hybrid, color='tab:green', label='Energy Hybrid', alpha=0.7)

    # Set labels and title
    ax_energy.set_xlabel('Elapsed Time (min)')
    ax_energy.set_ylabel('Energy (kWh)')
    ax_energy.set_title('Cumulative Energy Over Time')

    # Add legend
    ax_energy.legend(loc='upper right')

    # Optionally, set y-axis tick colors to default or customize as needed
    ax_energy.tick_params(axis='y', labelcolor='black')

    # Add subplot label
    ax_energy.text(-0.1, 1.05, labels[subplot_start + 3], transform=ax_energy.transAxes, fontsize=16, fontweight='bold',
                  va='bottom', ha='right')

# If there are remaining subplot axes (e.g., if less than 8 subplots), hide them
total_plots = len(csv_files) * 4
for i in range(total_plots, len(axes)):
    axes[i].axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig(save_path, dpi=300)

plt.show()
