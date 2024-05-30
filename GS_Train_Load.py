# %%
import os
import glob
import pandas as pd
import random
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from GS_vehicle_dict import vehicle_dict

# %%
def load_files(base_dir, vehicle_dict):
    all_files = []
    for vehicle, ids in vehicle_dict.items():
        for vid in ids:
            patterns = [
                os.path.join(base_dir, f"**/bms_{vid}-*"),
                os.path.join(base_dir, f"**/bms_altitude_{vid}-*")
            ]
            for pattern in patterns:
                all_files += glob.glob(pattern, recursive=True)
    return all_files

# %%
def add_predicted_power(files, model_path):
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    for file in files:
        try:
            data = pd.read_csv(file)
            if 'Power' in data.columns and 'Power_IV' in data.columns:
                features = data[['speed', 'acceleration']]
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                y_pred = model.predict(features_scaled)
                data['Predicted_Power'] = data['Power_IV'] - y_pred
                data.to_csv(file, index=False)
                print(f"Processed file {file}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")

# %%
def plot_comparison(data, file_name):
    data['time'] = pd.to_datetime(data['time'])
    interval = (data['time'].diff().dt.total_seconds().fillna(0) / 3600)
    Predicted_Energy = np.cumsum(data['Predicted_Power'] / 1000 * interval)
    Model_Energy = np.cumsum(data['Power'] / 1000 * interval)
    Actual_Energy = np.cumsum(data['Power_IV'] / 1000 * interval)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.set_title('Actual vs Predicted Power over Time')
    ax1.plot(data['time'], data['Power_IV'] / 1000, label='Actual Power_IV (kW)', color='blue')
    ax1.plot(data['time'], data['Predicted_Power'] / 1000, label='Predicted Power (kW)', color='red')
    ax1.plot(data['time'], data['Power'] / 1000, label='Model Power (kW)', color='green')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Power (kW)')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.95))
    ax1.grid(True)
    date = data['time'].iloc[0].strftime('%Y-%m-%d')
    ax1.text(0.99, 0.98, date, transform=ax1.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', color='black')
    ax1.text(0.01, 0.98, f'File: {file_name}', transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', color='black')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    ax2.set_title('Actual vs Predicted Energy over Time')
    ax2.plot(data['time'], Actual_Energy, label='Actual Energy (kWh)', color='blue')
    ax2.plot(data['time'], Predicted_Energy, label='Predicted Energy (kWh)', color='red')
    ax2.plot(data['time'], Model_Energy, label='Model Energy (kWh)', color='tab:green')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy (kWh)')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.95))
    ax2.grid(True)
    ax2.text(0.99, 0.98, date, transform=ax2.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', color='black')
    ax2.text(0.01, 0.98, f'File: {file_name}', transform=ax2.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', color='black')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    if 'altitude' in data.columns:
        data.set_index('time', inplace=True)
        data_resampled = data.resample('1T').mean()
        ax2_alt = ax2.twinx()
        ax2_alt.set_ylabel('Altitude (m)')
        ax2_alt.plot(data_resampled.index, data_resampled['altitude'], label='Altitude (m)', color='tab:orange', linestyle='-')
        ax2_alt.legend(loc='upper right', bbox_to_anchor=(1, 0.97))
        data.reset_index(inplace=True)
    
    plt.tight_layout()
    plt.show()


# %%
def main():
    base_dir = os.path.normpath(r'D:\SamsungSTF\Processed_Data\TripByTrip')
    model_path = os.path.join(os.path.dirname(base_dir), 'Power_model_XGBoost.json')


    all_files = load_files(base_dir, vehicle_dict)
    altitude_files = [file for file in all_files if 'altitude' in file]
    random.shuffle(altitude_files)
    
    for file in altitude_files[:5]:
        data = pd.read_csv(file)
        plot_comparison(data, os.path.basename(file).replace('.csv', ''))

if __name__ == "__main__":
    main()
