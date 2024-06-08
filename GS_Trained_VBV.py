import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from GS_vehicle_dict import vehicle_dict


def load_data_by_vehicle(base_dir, vehicle_dict, selected_vehicle):
    vehicle_files = {}
    if selected_vehicle not in vehicle_dict:
        print(f"Selected vehicle '{selected_vehicle}' not found in vehicle_dict.")
        return vehicle_files

    ids = vehicle_dict[selected_vehicle]
    all_files = []
    for vid in ids:
        patterns = [
            os.path.join(base_dir, f"**/bms_{vid}-*"),
            os.path.join(base_dir, f"**/bms_altitude_{vid}-*")
        ]
        for pattern in patterns:
            all_files += glob.glob(pattern, recursive=True)
    vehicle_files[selected_vehicle] = all_files

    return vehicle_files


def process_file(file, model, scaler):
    try:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns:
            original_data = data.copy()

            # Standardize speed and acceleration
            scaled_features = scaler.transform(data[['speed', 'acceleration']])

            dmatrix = xgb.DMatrix(scaled_features)
            predicted_residuals = model.predict(dmatrix)

            data['Predicted_Power'] = data['Power_IV'] - predicted_residuals

            # Save the updated file (overwriting the original file)
            data.to_csv(file, index=False)

            print(f"File saved with predicted power: {file}")
        else:
            print(f"File {file} does not contain required columns 'speed' and 'acceleration'.")
    except Exception as e:
        print(f"Error processing file {file}: {e}")


def add_predicted_power_column(files, model_path):
    try:
        model = xgb.Booster()
        model.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare a standard scaler based on all the data
    all_data = pd.concat([pd.read_csv(file)[['speed', 'acceleration']] for file in files if
                          'speed' in pd.read_csv(file).columns and 'acceleration' in pd.read_csv(file).columns])
    scaler = StandardScaler()
    scaler.fit(all_data)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, file, model, scaler) for file in files]
        for future in as_completed(futures):
            future.result()