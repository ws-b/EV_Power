import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import xgboost as xgb
from GS_vehicle_dict import vehicle_dict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.interpolate import griddata

def load_and_split_data_by_vehicle(base_dir, vehicle_dict, selected_vehicle):
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

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    full_data = pd.concat(df_list, ignore_index=True)

    scaler = StandardScaler()
    full_data[['speed', 'acceleration']] = scaler.fit_transform(full_data[['speed', 'acceleration']])

    # Debug: Check for out-of-range values after scaling
    print(f"Acceleration range after scaling: {full_data['acceleration'].min()} to {full_data['acceleration'].max()}")

    return full_data


def cross_validate(vehicle_files, selected_vehicle, n_splits=5, save_dir="models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    best_rmse = float('inf')
    best_model = None

    if selected_vehicle not in vehicle_files or not vehicle_files[selected_vehicle]:
        print(f"No files found for the selected vehicle: {selected_vehicle}")
        return

    files = vehicle_files[selected_vehicle]
    data = process_files(files)
    X = data[['speed', 'acceleration']].to_numpy()
    y = data['Residual'].to_numpy()
    groups = np.zeros(len(y))  # Dummy groups array as StratifiedKFold doesn't support group_k-fold directly

    for fold_num, (train_index, test_index) in enumerate(skf.split(X, groups), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': 'rmse'
        }
        evals = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(params, dtrain, num_boost_round=100, evals=evals)
        y_pred = model.predict(dtest)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append((fold_num, rmse))
        print(f"Vehicle: {selected_vehicle}, Fold: {fold_num}, RMSE: {rmse}")

        plot_3d(X_test, y_test, y_pred, fold_num, selected_vehicle)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    # Save the best model
    if best_model:
        model_file = os.path.join(save_dir, f"best_model_{selected_vehicle}.json")
        best_model.save_model(model_file)
        print(f"Best model for {selected_vehicle} saved with RMSE: {best_rmse}")

    return results

def plot_3d(X, y_true, y_pred, fold_num, vehicle):
    if X.shape[1] != 2:
        print("Error: X should have 2 columns.")
        return

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

    grid_x, grid_y = np.linspace(X[:, 0].min(), X[:, 0].max(), 100), np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
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
        title=f'3D Plot of Actual vs. Predicted Residuals (Fold {fold_num}, Vehicle: {vehicle})'
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
    save_dir = os.path.normpath(r'D:\SamsungSTF\Processed_Data\Models')

    selected_vehicle = 'NiroEV'  # 학습할 차종 지정

    vehicle_files = load_and_split_data_by_vehicle(base_dir, vehicle_dict, selected_vehicle)
    if not vehicle_files:
        print(f"No files found for the selected vehicle: {selected_vehicle}")
        return

    results = cross_validate(vehicle_files, selected_vehicle, save_dir=save_dir)

    # Print overall results
    if results:
        for fold_num, rmse in results:
            print(f"Fold: {fold_num}, RMSE: {rmse}")
    else:
        print(f"No results for the selected vehicle: {selected_vehicle}")

if __name__ == "__main__":
    main()