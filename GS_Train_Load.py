import os
import glob
import pandas as pd
import random
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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

def main():
    base_dir = os.path.normpath(r'D:\SamsungSTF\Processed_Data\TripByTrip')
    model_path = os.path.join(os.path.dirname(base_dir), 'Power_model_XGBoost.json')
    vehicle_dict = {
        'NiroEV': ['01241228149', '01241228151', '01241228153', '01241228154', '01241228155'],
        'Ionic5': ['01241227999', '01241228003', '01241228005', '01241228007', '01241228009', '01241228014',
                   '01241228016', '01241228020', '01241228024', '01241228025', '01241228026', '01241228030',
                   '01241228037', '01241228044', '01241228046', '01241228047', '01241248780', '01241248782',
                   '01241248790', '01241248811', '01241248815', '01241248817', '01241248820', '01241248827',
                   '01241364543', '01241364560', '01241364570', '01241364581', '01241592867', '01241592868',
                   '01241592878', '01241592896', '01241592907', '01241597801', '01241597802', '01241248919',
                   '01241321944'],
        'Ionic6': ['01241248713', '01241592904', '01241597763', '01241597804'],
        'KonaEV': ['01241228102', '01241228122', '01241228123', '01241228156', '01241228197', '01241228203', '01241228204',
                   '01241248726', '01241248727', '01241364621', '01241124056'],
        'EV6': ['01241225206', '01241228048', '01241228049', '01241228050', '01241228051', '01241228053', '01241228054',
                '01241228055', '01241228057', '01241228059', '01241228073', '01241228075', '01241228076', '01241228082',
                '01241228084', '01241228085', '01241228086', '01241228087', '01241228090', '01241228091', '01241228092',
                '01241228094', '01241228095', '01241228097', '01241228098', '01241228099', '01241228103', '01241228104',
                '01241228106', '01241228107', '01241228114', '01241228124', '01241228132', '01241228134', '01241248679',
                '01241248818', '01241248831', '01241248833', '01241248842', '01241248843', '01241248850', '01241248860',
                '01241248876', '01241248877', '01241248882', '01241248891', '01241248892', '01241248900', '01241248903',
                '01241248908', '01241248912', '01241248913', '01241248921', '01241248924', '01241248926', '01241248927',
                '01241248929', '01241248932', '01241248933', '01241248934', '01241321943', '01241321947', '01241364554',
                '01241364575', '01241364592', '01241364627', '01241364638', '01241364714'],
        'GV60': ['01241228108', '01241228130', '01241228131', '01241228136', '01241228137', '01241228138']
    }

    all_files = load_files(base_dir, vehicle_dict)
    altitude_files = [file for file in all_files if 'altitude' in file]
    random.shuffle(altitude_files)
    
    for file in altitude_files[:5]:
        data = pd.read_csv(file)
        plot_comparison(data, os.path.basename(file).replace('.csv', ''))

if __name__ == "__main__":
    main()
