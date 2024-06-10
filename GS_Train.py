import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def process_files(files):
    # Speed in m/s (160 km/h = 160 / 3.6 m/s)
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 180 / 3.6
    ACCELERATION_MIN = -15
    ACCELERATION_MAX = 9

    df_list = []
    for file in files:
        try:
            data = pd.read_csv(file)
            if 'Power' in data.columns and 'Power_IV' in data.columns:
                data['Residual'] = data['Power'] - data['Power_IV']
                data['speed'] = data['speed']
                df_list.append(data[['speed', 'acceleration', 'Residual']])
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    full_data = pd.concat(df_list, ignore_index=True)

    # Define scaler with the predefined range
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(pd.DataFrame([[SPEED_MIN, ACCELERATION_MIN], [SPEED_MAX, ACCELERATION_MAX]], columns=['speed', 'acceleration']))

    full_data[['speed', 'acceleration']] = scaler.transform(full_data[['speed', 'acceleration']])

    return full_data, scaler

def cross_validate(vehicle_files, selected_vehicle, save_dir="models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_rmse = float('inf')
    best_model = None

    if selected_vehicle not in vehicle_files or not vehicle_files[selected_vehicle]:
        print(f"No files found for the selected vehicle: {selected_vehicle}")
        return

    files = vehicle_files[selected_vehicle]
    data, scaler = process_files(files)
    X = data[['speed', 'acceleration']].to_numpy()
    y = data['Residual'].to_numpy()

    for fold_num, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': ['rmse', 'mae'],
            'objective': 'reg:squarederror'
        }
        evals = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(params, dtrain, num_boost_round=100, evals=evals)
        y_pred = model.predict(dtest)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append((fold_num, rmse))
        print(f"Vehicle: {selected_vehicle}, Fold: {fold_num}, RMSE: {rmse}")

        # plot_3d(X_test, y_test, y_pred, fold_num, selected_vehicle, scaler)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    # Save the best model
    if best_model:
        model_file = os.path.join(save_dir, f"best_model_{selected_vehicle}.json")
        best_model.save_model(model_file)
        print(f"Best model for {selected_vehicle} saved with RMSE: {best_rmse}")

    return results, scaler

def plot_3d(X, y_true, y_pred, fold_num, vehicle, scaler):
    if X.shape[1] != 2:
        print("Error: X should have 2 columns.")
        return

    # 역변환하여 원래 범위로 변환
    X_orig = scaler.inverse_transform(X)

    # Speed를 km/h로 변환
    X_orig[:, 0] *= 3.6

    sample_size = min(1000, X.shape[0])
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sampled = X_orig[sample_indices]
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

    grid_x, grid_y = np.linspace(X_orig[:, 0].min(), X_orig[:, 0].max(), 100), np.linspace(X_orig[:, 1].min(), X_orig[:, 1].max(), 100)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata((X_orig[:, 0], X_orig[:, 1]), y_pred, (grid_x, grid_y), method='linear')

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
            xaxis=dict(title='Speed (km/h)'),
            yaxis=dict(title='Acceleration (m/s²)'),
            zaxis=dict(title='Residual'),
        ),
        title=f'3D Plot of Actual vs. Predicted Residuals (Fold {fold_num}, Vehicle: {vehicle})'
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def process_file_with_trained_model(file, model, scaler):
    try:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns and 'Power_IV' in data.columns:
            # Use the provided scaler
            features = data[['speed', 'acceleration']]
            features_scaled = scaler.transform(features)

            predicted_residual = model.predict(features_scaled)

            data['Predicted_Power'] = data['Power'] - predicted_residual

            # Save the updated file
            data.to_csv(file, index=False)

            print(f"Processed file {file}")
        else:
            print(f"File {file} does not contain required columns 'speed', 'acceleration', or 'Power_IV'.")
    except Exception as e:
        print(f"Error processing file {file}: {e}")

def add_predicted_power_column(files, model_path, scaler):
    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_with_trained_model, file, model, scaler) for file in files]
        for future in as_completed(futures):
            future.result()
