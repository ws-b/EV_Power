import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from GS_Functions import calculate_rrmse, calculate_rmse, calculate_mape
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# ----------------------------
# Data Processing Functions
# ----------------------------
def process_single_file(file, files):
    """
    Process a single CSV file and extract relevant columns.
    """
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            data['trip_id'] = files.index(file)
            # Rolling statistics with window size 5
            data['mean_accel_10'] = data['acceleration'].rolling(window=5).mean().bfill()
            data['std_accel_10'] = data['acceleration'].rolling(window=5).std().bfill()
            data['mean_speed_10'] = data['speed'].rolling(window=5).mean().bfill()
            data['std_speed_10'] = data['speed'].rolling(window=5).std().bfill()
            return data[['time', 'speed', 'acceleration', 'ext_temp', 'Residual', 'Power_phys', 'Power_data', 'trip_id',
                         'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None

def process_files(files, scaler=None):
    """
    Process multiple CSV files in parallel, compute rolling statistics, and apply scaling.
    """
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6  # km/h to m/s conversion
    ACCELERATION_MIN = -15  # m/s^2
    ACCELERATION_MAX = 9    # m/s^2
    TEMP_MIN = -30
    TEMP_MAX = 50
    feature_cols = [
        'speed', 'acceleration', 'ext_temp',
        'mean_accel_10', 'std_accel_10',
        'mean_speed_10', 'std_speed_10'
    ]

    df_list = []
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file, files): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                data = future.result()
                if data is not None:
                    df_list.append(data)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    full_data = pd.concat(df_list, ignore_index=True)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame([
            [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN, 0, 0, 0, 0],
            [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX, 1, 1, 1, 1]
        ], columns=feature_cols))

    # Apply scaling to all features
    full_data[feature_cols] = scaler.transform(full_data[feature_cols])

    return full_data, scaler

# ----------------------------
# Model Training and Evaluation Functions
# ----------------------------
def train_model_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def integrate_and_compare(trip_data):
    """
    Integrate 'Power_hybrid' and 'Power_data' over time.
    """
    # Sort by 'time'
    trip_data = trip_data.sort_values(by='time')

    # Convert 'time' to seconds
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # Integrate 'Power_hybrid'
    hybrid_cum_integral = cumulative_trapezoid(trip_data['Power_hybrid'].values, time_seconds, initial=0)
    hybrid_integral = hybrid_cum_integral[-1]

    # Integrate 'Power_data'
    data_cum_integral = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)
    data_integral = data_cum_integral[-1]

    return hybrid_integral, data_integral

# ----------------------------
# Plotting Functions
# ----------------------------
def plot_errors_comparison(test_data, fold_num):
    """
    Plot instantaneous and cumulative errors for both models.
    """
    # test_trip_groups = test_data.groupby('trip_id')
    # 고유한 trip_id 추출 (데이터의 순서대로)
    trip_ids = test_data['trip_id'].unique()

    # 첫 20개의 trip_id 선택
    selected_trip_ids = trip_ids[:100]

    print(f"선택된 트립 ID들 (총 {len(selected_trip_ids)}개): {selected_trip_ids}")

    # 선택된 trip_id로 데이터 필터링
    selected_test_data = test_data[test_data['trip_id'].isin(selected_trip_ids)]

    # 그룹화
    test_trip_groups = selected_test_data.groupby('trip_id')

    for trip_id, group in test_trip_groups:
        group = group.sort_values(by='time')
        time_seconds = (group['time'] - group['time'].min()).dt.total_seconds().values

        # Plot instantaneous errors
        plt.figure(figsize=(14, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time_seconds, group['error_data_driven']/1000, label='ML(LR) ONLY Only Error', color='#FFA500', alpha=0.7)
        plt.plot(time_seconds, group['error_hybrid']/1000, label='Hybrid Model(LR) Error', color='#4682B4', alpha=0.7)
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f'Trip {trip_id} Instantaneous Error Comparison')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Error (kW)')
        plt.legend(loc='upper left')

        # Compute cumulative errors
        group['cumulative_error_data_driven'] = np.cumsum(group['error_data_driven'].values)/3600/1000
        group['cumulative_error_hybrid'] = np.cumsum(group['error_hybrid'].values)/3600/1000

        # Plot cumulative errors
        plt.subplot(2, 1, 2)
        plt.plot(time_seconds, group['cumulative_error_data_driven'], label='ML(LR) ONLY Cumulative Error', color='#FFA500', alpha=0.7)
        plt.plot(time_seconds, group['cumulative_error_hybrid'], label='Hybrid Model(LR) Cumulative Error', color='#4682B4', alpha=0.7)
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f'Trip {trip_id} Cumulative Error Comparison')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Cumulative Error (kWh)')
        plt.legend(loc='upper left')

        plt.tight_layout()

        # 저장
        save_path = os.path.join(r"C:\Users\BSL\Desktop\Figures\Supplementary\FigureS5", f"figure_S5_{trip_id}.png")
        plt.savefig(save_path, dpi=300)
        print(f"저장됨: {save_path}")

        plt.show()

# ----------------------------
# Cross-Validation Function
# ----------------------------
def cross_validate_models(vehicle_files, selected_car):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]

    fold_num = 1
    for train_index, test_index in kf.split(files):
        print(f"Processing Fold {fold_num}")

        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        # Process training and test data
        train_data, scaler = process_files(train_files)
        test_data, _ = process_files(test_files, scaler=scaler)

        feature_cols = [
            'speed', 'acceleration', 'ext_temp',
            'mean_accel_10', 'std_accel_10',
            'mean_speed_10', 'std_speed_10'
        ]

        # Prepare training and test data for both models
        X_train = train_data[feature_cols]
        y_train_power_data = train_data['Power_data']
        y_train_residual = train_data['Residual']  # For hybrid model

        X_test = test_data[feature_cols]
        y_test_power_data = test_data['Power_data']
        y_test_residual = test_data['Residual']  # For hybrid model

        # Initialize models
        data_driven_model = LinearRegression()
        hybrid_model = LinearRegression()

        # Train models
        data_driven_model.fit(X_train, y_train_power_data)
        hybrid_model.fit(X_train, y_train_residual)

        # Make predictions
        test_data['y_pred_data_driven'] = data_driven_model.predict(X_test)
        test_data['y_pred_hybrid_residual'] = hybrid_model.predict(X_test)
        test_data['y_pred_hybrid'] = test_data['Power_phys'] + test_data['y_pred_hybrid_residual']
        test_data['Power_hybrid'] = test_data['y_pred_hybrid']

        # Compute errors
        test_data['error_data_driven'] = test_data['y_pred_data_driven'] - y_test_power_data
        test_data['error_hybrid'] = test_data['y_pred_hybrid'] - y_test_power_data

        # Compute RMSE
        rmse_data_driven = calculate_rmse(y_test_power_data, test_data['y_pred_data_driven'])
        rmse_hybrid = calculate_rmse(y_test_power_data, test_data['y_pred_hybrid'])

        # Compute Total Energy Error
        test_trip_groups = test_data.groupby('trip_id')
        total_energy_error_data_driven = []
        total_energy_error_hybrid = []
        for trip_id, group in test_trip_groups:
            group = group.sort_values(by='time')
            time_seconds = (group['time'] - group['time'].min()).dt.total_seconds().values

            # Integrate errors over time using the trapezoidal rule
            total_error_data_driven = np.trapz(group['error_data_driven'].values, time_seconds)
            total_error_hybrid = np.trapz(group['error_hybrid'].values, time_seconds)

            total_energy_error_data_driven.append(total_error_data_driven)
            total_energy_error_hybrid.append(total_error_hybrid)

        avg_total_energy_error_data_driven = np.mean(total_energy_error_data_driven)
        avg_total_energy_error_hybrid = np.mean(total_energy_error_hybrid)

        # Store results
        results.append({
            'fold': fold_num,
            'rmse_data_driven': rmse_data_driven,
            'rmse_hybrid': rmse_hybrid,
            'avg_total_energy_error_data_driven': avg_total_energy_error_data_driven,
            'avg_total_energy_error_hybrid': avg_total_energy_error_hybrid
        })

        # Plot errors and cumulative errors for the first fold
        if fold_num == 1:
            plot_errors_comparison(test_data, fold_num)

        # Print fold results
        print(f"--- Fold {fold_num} Results ---")
        print(f"RMSE (Data-Driven Model): {rmse_data_driven:.2f} W")
        print(f"RMSE (Hybrid Model): {rmse_hybrid:.2f} W")
        print(f"Average Total Energy Error (Data-Driven): {avg_total_energy_error_data_driven:.2f} W*s")
        print(f"Average Total Energy Error (Hybrid): {avg_total_energy_error_hybrid:.2f} W*s")
        print("---------------------------\n")

        # Increment fold number
        fold_num += 1

    # After cross-validation, you can compute average metrics or select the best model based on RMSE
    return results