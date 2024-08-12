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


def calculate_rrmse(y_test, y_pred):
    relative_errors = (y_pred - y_test) / np.mean(y_test)
    rrmse = np.sqrt(np.mean(relative_errors ** 2))
    return rrmse

def calculate_rmse(y_test, y_pred):
    rmse = np.sqrt(np.mean((y_test-y_pred) ** 2))
    return rmse

def read_and_process_files(files):
    data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    return data

def compute_rrmse(vehicle_files, selected_car):
    if not vehicle_files:
        print("No files provided")
        return

    data = read_and_process_files(vehicle_files[selected_car])

    if 'Power' not in data.columns or 'Power_IV' not in data.columns:
        print(f"Columns 'Power' and/or 'Power_IV' not found in the data")
        return

    y_pred = data['Power'].to_numpy()
    y_test = data['Power_IV'].to_numpy()

    rrmse = calculate_rrmse(y_test, y_pred)
    print(f"RRMSE for {selected_car}  : {rrmse}")
    return rrmse

def compute_rmse(vehicle_files, selected_car):
    if not vehicle_files:
        print("No files provided")
        return

    data = read_and_process_files(vehicle_files[selected_car])

    if 'Power' not in data.columns or 'Power_IV' not in data.columns:
        print(f"Columns 'Power' and/or 'Power_IV' not found in the data")
        return

    y_pred = data['Power'].to_numpy()
    y_test = data['Power_IV'].to_numpy()

    rmse = calculate_rmse(y_test, y_pred)
    print(f"RMSE for {selected_car}  : {rmse}")
    return rmse
