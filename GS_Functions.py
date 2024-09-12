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
    rrmse = np.sqrt(np.mean(relative_errors ** 2)) * 100
    return rrmse

def calculate_rmse(y_test, y_pred):
    rmse = np.sqrt(np.mean((y_test-y_pred) ** 2))
    return rmse

def integrate_and_compare(trip_data):
    trip_data['time'] = pd.to_datetime(trip_data['time'], format='%Y-%m-%d %H:%M:%S')
    trip_data = trip_data.sort_values(by='time')

    # 'time'을 초 단위로 변환
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    physics_integral = np.trapz(trip_data['Power_phys'].values, time_seconds)
    data_integral = np.trapz(trip_data['Power_data'].values, time_seconds)

    # 적분된 값 반환
    return physics_integral, data_integral
def compute_mape_rrmse(vehicle_files, selected_car):
    if not vehicle_files:
        print("No files provided")
        return
    physics_integrals, data_integrals = [], []
    for file in vehicle_files[selected_car]:
        data = pd.read_csv(file)
        physics_integral, data_integral = integrate_and_compare(data)
        physics_integrals.append(physics_integral)
        data_integrals.append(data_integral)

    mape= calculate_mape(np.array(data_integrals), np.array(physics_integrals))
    rrmse = calculate_rrmse(np.array(data_integrals), np.array(physics_integrals))
    print(f"MAPE for {selected_car}  : {mape:.2f}%, RRMSE for {selected_car}: {rrmse:.2f}%")

    return mape, rrmse