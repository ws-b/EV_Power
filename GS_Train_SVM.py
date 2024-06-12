import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from GS_plot import plot_3d
from concurrent.futures import ProcessPoolExecutor, as_completed

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

    y_mean = np.mean(y)
    y_range = np.ptp(y)  # range of the residuals

    for fold_num, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        nrmse = rmse / y_range
        percent_rmse = (rmse / y_mean) * 100
        results.append((fold_num, rmse, nrmse, percent_rmse))
        print(f"Vehicle: {selected_vehicle}, Fold: {fold_num}, RMSE: {rmse}, NRMSE: {nrmse}, Percent RMSE: {percent_rmse}%")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    # Save the best model
    if best_model:
        model_file = os.path.join(save_dir, f"SVM_best_model_{selected_vehicle}.pkl")
        surface_plot = os.path.join(save_dir, f"SVM_best_model_{selected_vehicle}_plot.html")
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Best model for {selected_vehicle} saved with RMSE: {best_rmse}")
        plot_3d(X_test, y_test, y_pred, fold_num, selected_vehicle, scaler, 400, 30,
                output_file=surface_plot)
    return results, scaler

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
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_with_trained_model, file, model, scaler) for file in files]
        for future in as_completed(futures):
            future.result()
