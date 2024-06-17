import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GS_vehicle_dict import vehicle_dict
from concurrent.futures import ThreadPoolExecutor

# Directory containing CSV files
directory = r"D:\SamsungSTF\Processed_Data\TripByTrip"

def process_file(file_path):
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)
        device_id = os.path.basename(file_path).split('-')[1]
        # Convert speed from m/s to km/h
        data['speed'] = data['speed'] * 3.6
        return device_id, data['speed'], data['acceleration']
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None, None


def sample_grid(speed, acceleration, grid_size=100, max_per_grid=30):
    grid = {}
    min_speed, max_speed = speed.min(), speed.max()
    min_accel, max_accel = acceleration.min(), acceleration.max()

    speed_bins = np.linspace(min_speed, max_speed, grid_size + 1)
    accel_bins = np.linspace(min_accel, max_accel, grid_size + 1)

    for s, a in zip(speed, acceleration):
        s_bin = np.digitize(s, speed_bins) - 1
        a_bin = np.digitize(a, accel_bins) - 1
        grid_key = (s_bin, a_bin)
        if grid_key not in grid:
            grid[grid_key] = []
        if len(grid[grid_key]) < max_per_grid:
            grid[grid_key].append((s, a))

    sampled_speeds = []
    sampled_accelerations = []
    for points in grid.values():
        sampled_speeds.extend([p[0] for p in points])
        sampled_accelerations.extend([p[1] for p in points])

    return sampled_speeds, sampled_accelerations


# Iterate through each vehicle type in the vehicle_dict
for vehicle_type, device_ids in vehicle_dict.items():
    all_speeds = []
    all_accelerations = []

    # Get the list of relevant files
    relevant_files = [os.path.join(directory, f) for f in os.listdir(directory) if
                      f.endswith(".csv") and any(device_id in f for device_id in device_ids)]

    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, relevant_files)

        for device_id, speeds, accelerations in results:
            if speeds is not None and accelerations is not None:
                all_speeds.extend(speeds)
                all_accelerations.extend(accelerations)

    # Convert to numpy arrays for easy handling
    all_speeds = np.array(all_speeds)
    all_accelerations = np.array(all_accelerations)

    # Sample the data in the grids
    sampled_speeds, sampled_accelerations = sample_grid(all_speeds, all_accelerations)

    # Create a scatter plot for the current vehicle type
    plt.figure(figsize=(10, 6))
    plt.scatter(sampled_speeds, sampled_accelerations, alpha=0.5)
    plt.title(f'Speed vs Acceleration for {vehicle_type}')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Acceleration (m/s²)')
    plt.xlim(0, 230)
    plt.ylim(-15, 9)
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(os.path.join(directory, f'{vehicle_type}_speed_vs_acceleration.png'))
    plt.close()

print("Plots generated and saved successfully.")