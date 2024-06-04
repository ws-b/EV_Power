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
from GS_vehicle_dict import vehicle_dict

def load_and_split_data(base_dir, vehicle_dict, train_count=500, test_count=120, test_ratio=0.2):
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
    else:
        train_files = all_files[:train_count]
        test_files = all_files[train_count:train_count + test_count]
    return train_files, test_files


def process_files(files):
    df_list = []
    out_of_range_files = []
    for file in files:
        try:
            data = pd.read_csv(file)
            if 'Power' in data.columns and 'Power_IV' in data.columns:
                if data['acceleration'].abs().max() > 9.8:
                    out_of_range_files.append(file)
                data['Residual'] = data['Power_IV'] - data['Power']
                df_list.append(data[['speed', 'acceleration', 'Residual']])
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    if out_of_range_files:
        print("Out-of-range acceleration values detected in the following files:")
        for f in out_of_range_files:
            print(f)
    full_data = pd.concat(df_list, ignore_index=True)
    scaler = StandardScaler()
    full_data[['speed', 'acceleration']] = scaler.fit_transform(full_data[['speed', 'acceleration']])

    # Debug: Check for out-of-range values after scaling
    print(f"Acceleration range after scaling: {full_data['acceleration'].min()} to {full_data['acceleration'].max()}")

    return full_data

def plot_3d(X, y_true, y_pred):
    if X.shape[1] != 2:
        print("Error: X should have 2 columns.")
        return
    assert X[:, 1].min() >= -9.8 and X[:, 1].max() <= 9.8, "Acceleration values out of range detected!"
    sample_size = min(1000, X.shape[0])
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sampled = X[sample_indices]
    y_true_sampled = y_true[sample_indices]
    y_pred_sampled = y_pred[sample_indices]
    trace1 = go.Scatter3d(
        x=X_sampled[:, 0], y=X_sampled[:, 1], z=y_true_sampled,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.8),
        name='Actual Residual'
    )
    trace2 = go.Scatter3d(
        x=X_sampled[:, 0], y=X_sampled[:, 1], z=y_pred_sampled,
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.8),
        name='Predicted Residual'
    )
    grid_x, grid_y = np.linspace(min(X[:, 0]), max(X[:, 0]), 100), np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata((X[:, 0], X[:, 1]), y_pred, (grid_x, grid_y), method='linear')
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
    ax2.plot(data['time'], Model_Energy, label='Model Energy (kWh)', color='green')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy (kWh)')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.95))
    ax2.grid(True)
    ax2.text(0.99, 0.98, date, transform(ax2.transAxes, fontsize=12, verticalalignment='top',
                                         horizontalalignment='right', color='black'))
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

    train_files, test_files = load_and_split_data(base_dir, vehicle_dict)
    train_data = process_files(train_files)
    test_data = process_files(test_files)

    # Debug: Check for out-of-range values in test data
    print(f"Acceleration range in test data: {test_data['acceleration'].min()} to {test_data['acceleration'].max()}")

    X_train = train_data[['speed', 'acceleration']].to_numpy()
    y_train = train_data['Residual'].to_numpy()
    X_test = test_data[['speed', 'acceleration']].to_numpy()
    y_test = test_data['Residual'].to_numpy()
    base_margin_train = train_data['speed'].to_numpy()
    base_margin_test = test_data['speed'].to_numpy()
    dtrain = xgb.DMatrix(X_train, label=y_train, base_margin=base_margin_train)
    dtest = xgb.DMatrix(X_test, label=y_test, base_margin=base_margin_test)
    params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'rmse'
    }
    evals = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, obj=custom_objective)
    y_pred = model.predict(dtest)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE for Residual Prediction: {rmse}")
    plot_3d(X_test, y_test, y_pred)
    for i in range(min(9, len(test_files))):
        specific_file = test_files[i]
        specific_file_name = os.path.basename(specific_file)
        specific_data = pd.read_csv(specific_file)
        specific_features = specific_data[['speed', 'acceleration']]
        scaler = StandardScaler()
        specific_features_scaled = scaler.fit_transform(specific_features)
        specific_dtest = xgb.DMatrix(np.column_stack((specific_data['speed'].to_numpy(), specific_features_scaled)))
        specific_y_pred = model.predict(specific_dtest)
        plot_full_time_series(specific_data, specific_y_pred, specific_file_name)

    """
    # Save model
    parent_dir = os.path.dirname(base_dir)
    model.save_model(os.path.join(parent_dir, 'Power_model_XGBoost.json'))
    """

if __name__ == "__main__":
    main()
