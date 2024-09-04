import os
import pandas as pd
import numpy as np
from GS_preprocessing import load_data_by_vehicle
def get_vehicle_files(car_options, folder_path, vehicle_dict):
    selected_cars = []
    vehicle_files = {}
    while True:
        print("Available Cars:")
        for idx, car_name in car_options.items():
            print(f"{idx}: {car_name}")
        print("0: Done selecting cars")
        car_input = input("Select Cars you want to include 콤마로 구분 (e.g.: 1,2,3): ")

        try:
            car_list = [int(car.strip()) for car in car_input.split(',')]
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
            continue

        if 0 in car_list:
            car_list.remove(0)
            for car in car_list:
                if car in car_options:
                    selected_car = car_options[car]
                    if selected_car not in selected_cars:
                        selected_cars.append(selected_car)
                        vehicle_files = vehicle_files | load_data_by_vehicle(folder_path, vehicle_dict, selected_car)
                else:
                    print(f"Invalid choice: {car}. Please try again.")
            break
        else:
            for car in car_list:
                if car in car_options:
                    selected_car = car_options[car]
                    if selected_car not in selected_cars:
                        selected_cars.append(selected_car)
                        vehicle_files= vehicle_files | load_data_by_vehicle(folder_path, vehicle_dict, selected_car)
                else:
                    print(f"Invalid choice: {car}. Please try again.")

    return selected_cars, vehicle_files
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


def calculate_mape(y_test, y_pred):
    # y_test가 0이 아닌 값들만 선택하여 MAPE 계산
    non_zero_indices = y_test != 0
    y_test_non_zero = y_test[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]

    mape = np.mean(np.abs((y_test_non_zero - y_pred_non_zero) / y_test_non_zero)) * 100
    return mape
def calculate_rrmse(y_test, y_pred):
    relative_errors = (y_test - y_pred) / np.mean(np.abs(y_test))
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

    if 'Power_phys' not in data.columns or 'Power_data' not in data.columns:
        print(f"Columns 'Power_phys' and/or 'Power_data' not found in the data")
        return

    y_pred = data['Power_phys'].to_numpy()
    y_test = data['Power_data'].to_numpy()

    # data['time'] = pd.to_datetime(data['time'])
    #
    # data['minute'] = data['time'].dt.floor('min')
    # grouped = data.groupby('minute')
    #
    # y_test_integrated = grouped.apply(lambda group: np.trapz(group['Power_data'], x=group['time'].astype('int64') / 1e9))
    # y_pred_integrated = grouped.apply(lambda group: np.trapz(group['Power_phys'], x=group['time'].astype('int64') / 1e9))
    #
    # rrmse = calculate_rrmse(y_test_integrated, y_pred_integrated)
    rrmse = calculate_rrmse(y_test, y_pred)
    print(f"RRMSE for {selected_car}  : {rrmse}")
    return rrmse

def compute_rmse(vehicle_files, selected_car):
    if not vehicle_files:
        print("No files provided")
        return

    data = read_and_process_files(vehicle_files[selected_car])

    if 'Power_phys' not in data.columns or 'Power_data' not in data.columns:
        print(f"Columns 'Power_phys' and/or 'Power_data' not found in the data")
        return

    y_pred = data['Power_phys'].to_numpy()
    y_test = data['Power_data'].to_numpy()

    # data['time'] = pd.to_datetime(data['time'])
    #
    # data['minute'] = data['time'].dt.floor('min')
    # grouped = data.groupby('minute')
    #
    # y_test_integrated = grouped.apply(lambda group: np.trapz(group['Power_data'], x=group['time'].astype('int64') / 1e9))
    # y_pred_integrated = grouped.apply(lambda group: np.trapz(group['Power_phys'], x=group['time'].astype('int64') / 1e9))
    #
    # rmse = calculate_rmse(y_test_integrated, y_pred_integrated)
    rmse = calculate_rmse(y_test, y_pred)
    print(f"RMSE for {selected_car}  : {rmse}")
    return rmse

def compute_mape(vehicle_files, selected_car):
    if not vehicle_files:
        print("No files provided")
        return

    data = read_and_process_files(vehicle_files[selected_car])

    if 'Power_phys' not in data.columns or 'Power_data' not in data.columns:
        print(f"Columns 'Power_phys' and/or 'Power_data' not found in the data")
        return

    y_pred = data['Power_phys'].to_numpy()
    y_test = data['Power_data'].to_numpy()

    # data['time'] = pd.to_datetime(data['time'])
    #
    # data['minute'] = data['time'].dt.floor('min')
    # grouped = data.groupby('minute')
    #
    # y_test_integrated = grouped.apply(lambda group: np.trapz(group['Power_data'], x=group['time'].astype('int64') / 1e9))
    # y_pred_integrated = grouped.apply(lambda group: np.trapz(group['Power_phys'], x=group['time'].astype('int64') / 1e9))
    #
    # mape = calculate_mape(y_test_integrated, y_pred_integrated)
    mape = calculate_mape(y_test, y_pred)
    print(f"MAPE for {selected_car}  : {mape}%")
    return mape

def add_rush_hour_and_weekend_feature(data):
    date_formats = ['%Y-%m-%d %H:%M:%S', '%y-%m-%d %H:%M:%S']
    for date_format in date_formats:
        try:
            # Parse the date using the current format
            data['time'] = pd.to_datetime(data['time'], format=date_format)
            # Extract the hour and weekday
            data['hour'] = data['time'].dt.hour
            data['weekday'] = data['time'].dt.weekday  # Monday=0, Sunday=6
            break
        except ValueError as e:
            print(f"Date format error with format {date_format}: {e}")
            continue
    else:
        print("None of the provided date formats matched the data")
        return data

    # Define weekend and weekdays
    data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)  # Saturday=5, Sunday=6
    data['is_weekday'] = (~data['is_weekend']).astype(int)

    # Define rush hour periods for weekdays only
    rush_hour_morning = (data['hour'] >= 6) & (data['hour'] <= 9)
    rush_hour_evening = (data['hour'] >= 17) & (data['hour'] <= 20)
    data['is_rush_hour'] = (rush_hour_morning | rush_hour_evening) & (data['is_weekday'] == 1)
    data['is_rush_hour'] = data['is_rush_hour'].astype(int)  # Convert to 0 or 1

    # Optionally drop the 'hour' and 'weekday' columns if no longer needed
    data.drop(columns=['hour', 'weekday'], inplace=True)

    return data

