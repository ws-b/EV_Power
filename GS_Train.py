import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import griddata

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

def process_file_with_trained_model(file, model, scaler):
    try:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns:
            # Standardize speed and acceleration
            scaled_features = scaler.transform(data[['speed', 'acceleration']])

            dmatrix = xgb.DMatrix(scaled_features)
            predicted_residuals = model.predict(dmatrix)

            data['Predicted_Power'] = data['Power'] - predicted_residuals

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
        futures = [executor.submit(process_file_with_trained_model, file, model, scaler) for file in files]
        for future in as_completed(futures):
            future.result()