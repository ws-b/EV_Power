import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata


def load_and_split_data(base_dir, vehicle_dict, train_count=4000, test_count=1000, test_ratio=0.2):
    all_files = []

    for vehicle, ids in vehicle_dict.items():
        for vid in ids:
            patterns = [
                os.path.join(base_dir, f"**/bms_{vid}-*"),
                os.path.join(base_dir, f"**/bms_altitude_{vid}-*")
            ]
            for pattern in patterns:
                all_files += glob.glob(pattern, recursive=True)

    random.shuffle(all_files)

    if test_ratio is not None:
        split_index = int(len(all_files) * (1 - test_ratio))
        train_files = all_files[:split_index]
        test_files = all_files[split_index:]
        print(int(len(all_files)))
    else:
        train_files = all_files[:train_count]
        test_files = all_files[train_count:train_count + test_count]

    return train_files, test_files


def process_files(files):
    df_list = []
    for file in files:
        try:
            data = pd.read_csv(file)
            if 'Power' in data.columns and 'Power_IV' in data.columns:
                data['Residual'] = data['Power_IV'] - data['Power']
                df_list.append(data[['speed', 'acceleration', 'Residual']])
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    full_data = pd.concat(df_list, ignore_index=True)
    scaler = StandardScaler()
    full_data[['speed', 'acceleration']] = scaler.fit_transform(full_data[['speed', 'acceleration']])
    return full_data


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + np.finfo(float).eps))) * 100


def plot_3d(X, y_true, y_pred):
    # Check data shapes and types
    print(f"X shape: {X.shape}, y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

    # Create the scatter plot for actual residuals
    trace1 = go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=y_true,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.8),
        name='Actual Residual'
    )

    # Create the scatter plot for predicted residuals
    trace2 = go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=y_pred,
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.8),
        name='Predicted Residual'
    )

    # Generate a grid for surface plot
    grid_x, grid_y = np.linspace(min(X[:, 0]), max(X[:, 0]), 100), np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Perform griddata interpolation
    grid_z = griddata((X[:, 0], X[:, 1]), y_pred, (grid_x, grid_y), method='nearest')

    # Check for NaN values in the interpolated grid
    if np.isnan(grid_z).any():
        print("Warning: NaN values detected in grid_z after interpolation.")
        # Fill NaN values using nearest neighbor interpolation
        mask = np.isnan(grid_z)
        grid_z[mask] = griddata((X[:, 0], X[:, 1]), y_pred, (grid_x[mask], grid_y[mask]), method='nearest')

    # Create a surface plot for predicted residuals
    surface_trace = go.Surface(
        x=grid_x,
        y=grid_y,
        z=grid_z,
        colorscale='Viridis',
        name='Predicted Residual Surface',
        opacity=0.7
    )

    data = [trace1, trace2, surface_trace]
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='Speed'),
            yaxis=dict(title='Acceleration'),
            zaxis=dict(title='Residual'),
        ),
        title='3D Plot of Actual vs. Predicted Residuals'
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def plot_full_time_series(data, y_pred, file_name):
    data['time'] = pd.to_datetime(data['time'])
    predicted_power = data['Power_IV'] - y_pred
    interval = (data['time'].diff().dt.total_seconds().fillna(0) / 3600)
    Actual_Energy = np.cumsum(data['Power_IV'] / 1000 * interval)
    Model_Energy = np.cumsum(data['Power'] / 1000 * interval)
    Predicted_Energy = np.cumsum(predicted_power / 1000 * interval)

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
    ax1.text(0.99, 0.98, date, transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             horizontalalignment='right', color='black')
    ax1.text(0.01, 0.98, f'File: {file_name}', transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='left', color='black')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    ax2.set_title('Actual vs Predicted Energy over Time')
    ax2.plot(data['time'], Actual_Energy, label='Actual Energy (kWh)', color='blue')
    ax2.plot(data['time'], Predicted_Energy, label='Predicted Energy (kWh)', color='red')
    ax2.plot(data['time'], Model_Energy, label='Model Energy (kWh)', color='tab:green')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy (kWh)')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.95))
    ax2.grid(True)
    ax2.text(0.99, 0.98, date, transform=ax2.transAxes, fontsize=12, verticalalignment='top',
             horizontalalignment='right', color='black')
    ax2.text(0.01, 0.98, f'File: {file_name}', transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='left', color='black')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    if 'altitude' in data.columns:
        data.set_index('time', inplace=True)
        data_resampled = data.resample('1T').mean()
        ax2_alt = ax2.twinx()
        ax2_alt.set_ylabel('Altitude (m)')
        ax2_alt.plot(data_resampled.index, data_resampled['altitude'], label='Altitude (m)', color='tab:orange',
                     linestyle='-')
        ax2_alt.legend(loc='upper right', bbox_to_anchor=(1, 0.97))
        data.reset_index(inplace=True)

    plt.tight_layout()
    plt.show()


def main():
    base_dir = os.path.normpath(r'D:\SamsungSTF\Processed_Data\TripByTrip')
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
        'KonaEV': ['01241228102', '01241228122', '01241228123', '01241228156', '01241228197', '01241228203',
                   '01241228204',
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

    train_files, test_files = load_and_split_data(base_dir, vehicle_dict)
    train_data = process_files(train_files)
    test_data = process_files(test_files)

    # Prepare data for XGBoost
    X_train = train_data[['speed', 'acceleration']].to_numpy()
    y_train = train_data['Residual'].to_numpy()
    X_test = test_data[['speed', 'acceleration']].to_numpy()
    y_test = test_data['Residual'].to_numpy()

    # Initialize DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost model parameters
    params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'rmse',
        'objective': 'reg:squarederror'
    }

    # Training with GPU
    evals = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(params, dtrain, num_boost_round=100, evals=evals)

    # Predictions
    y_pred = model.predict(dtest)

    # Calculate MAPE, RMSE
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"Test MAPE for Residual Prediction: {mape:.2f}%")

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE for Residual Prediction: {rmse}")

    # Plot results
    plot_3d(X_test, y_test, y_pred)

    # Plot results for a specific file
    for i in range(9):
        specific_file = test_files[i]
        specific_file_name = os.path.basename(specific_file)
        specific_data = pd.read_csv(specific_file)
        specific_features = specific_data[['speed', 'acceleration']]
        scaler = StandardScaler()
        specific_features_scaled = scaler.fit_transform(specific_features)
        specific_dtest = xgb.DMatrix(specific_features_scaled)
        specific_y_pred = model.predict(specific_dtest)
        plot_full_time_series(specific_data, specific_y_pred, specific_file_name)

    # Save model
    parent_dir = os.path.dirname(base_dir)
    model.save_model(os.path.join(parent_dir, 'Power_model_XGBoost.json'))


if __name__ == "__main__":
    main()