import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from GS_Functions import calculate_rrmse, calculate_rmse, calculate_mape
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from GS_plot import plot_contour, plot_shap_values
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_file(file):
    """
    Processes a single CSV file to calculate residuals and select relevant columns.
    """
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            return data[['time', 'speed', 'acceleration', 'ext_temp', 'Residual', 'Power_phys', 'Power_data', 'slope']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None

def process_files(files, scaler=None):
    """
    Processes multiple CSV files in parallel, calculates rolling statistics, and scales features.
    """
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6  # Convert 230 km/h to m/s
    ACCELERATION_MIN = -15  # m/s^2
    ACCELERATION_MAX = 9    # m/s^2
    TEMP_MIN = -30
    TEMP_MAX = 50
    feature_cols = ['speed', 'acceleration', 'ext_temp', 'mean_accel_10', 'std_accel_10',
                   'mean_speed_10', 'std_speed_10']

    df_list = []
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                data = future.result()
                if data is not None:
                    # Convert 'time' column to datetime
                    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

                    # Add trip_id for trip differentiation
                    data['trip_id'] = files.index(file)

                    # Calculate rolling statistics with a window of 5
                    data['mean_accel_10'] = data['acceleration'].rolling(window=5).mean().bfill()
                    data['std_accel_10'] = data['acceleration'].rolling(window=5).std().bfill()
                    data['mean_speed_10'] = data['speed'].rolling(window=5).mean().bfill()
                    data['std_speed_10'] = data['speed'].rolling(window=5).std().bfill()

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

def integrate_and_compare(trip_data):
    """
    Integrates 'Power_hybrid' and 'Power_data' over time using the trapezoidal rule.
    """
    # Sort by 'time'
    trip_data = trip_data.sort_values(by='time')

    # Convert 'time' to seconds
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # Integrate 'Power_phys + y_pred' using trapezoidal rule
    trip_data['Power_hybrid'] = trip_data['Power_phys'] + trip_data['y_pred']
    hybrid_integral = np.trapz(trip_data['Power_hybrid'].values, time_seconds)

    # Integrate 'Power_data' using trapezoidal rule
    data_integral = np.trapz(trip_data['Power_data'].values, time_seconds)

    return hybrid_integral, data_integral

def grid_search_lambda(X_train, y_train):
    """
    Performs GridSearchCV to find the optimal 'lambda_' (L2 regularization term) for XGBRegressor.
    """
    # Define the range of lambda values to search
    lambda_values = np.logspace(-5, 7, num=13)

    # Define the parameter grid with 'lambda_' as a list
    param_grid = {
        'lambda_': lambda_values  # Note the trailing underscore
    }

    # Instantiate the XGBRegressor with fixed parameters
    model = xgb.XGBRegressor(
        tree_method='gpu_hist',      # Use GPU-accelerated tree method
        gpu_id=0,                    # Specify the GPU ID
        eval_metric='rmse',          # Evaluation metric
        use_label_encoder=False,     # To avoid a warning in newer XGBoost versions
        verbosity=0                  # Suppress training logs
    )

    # Initialize GridSearchCV with the defined model and parameter grid
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1  # Utilize all available CPU cores for parallelism
    )

    # Fit the GridSearchCV to find the best lambda
    grid_search.fit(X_train, y_train)

    # Extract the best lambda value
    best_lambda = grid_search.best_params_['lambda_']
    print(f"Best lambda found: {best_lambda}")

    return best_lambda

def cross_validate(vehicle_files, selected_car, precomputed_lambda, plot=None, save_dir="models"):
    """
    Performs cross-validation with KFold and trains XGBRegressor models.
    """
    model_name = "XGB"

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []
    best_model = None

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]
    # Filter files containing 'bms' and 'altitude' in their filenames
    files = [file for file in vehicle_files[selected_car] if 'bms' in file.lower() and 'altitude' in file.lower()]

    best_lambda = precomputed_lambda
    for fold_num, (train_index, test_index) in enumerate(kf.split(files), 1):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        # Process train and test data
        train_data, scaler = process_files(train_files)
        test_data, _ = process_files(test_files)

        feature_cols = ['speed', 'acceleration', 'ext_temp', 'mean_accel_10', 'std_accel_10',
                       'mean_speed_10', 'std_speed_10']

        # Prepare training and testing data
        X_train = train_data[feature_cols]
        y_train = train_data['Residual']

        X_test = test_data[feature_cols]
        y_test = test_data['Residual']

        # Perform Grid Search if lambda is not precomputed
        if best_lambda is None:
            best_lambda = grid_search_lambda(X_train, y_train)

        # Instantiate the XGBRegressor with the best lambda and fixed parameters
        model = xgb.XGBRegressor(
            tree_method='gpu_hist',
            gpu_id=0,
            eval_metric='rmse',
            use_label_encoder=False,
            verbosity=0,
            lambda_=best_lambda,
            eta=0.3,
            n_estimators=150  # Corresponds to num_boost_round
        )

        # Fit the model with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )

        # Predict on train and test
        train_data['y_pred'] = model.predict(X_train)
        test_data['y_pred'] = model.predict(X_test)

        if plot:
            # Calculate and plot SHAP values
            plot_shap_values(model, X_train, feature_cols)

        # Group by trip_id and calculate integrals
        train_trip_groups = train_data.groupby('trip_id')
        test_trip_groups = test_data.groupby('trip_id')

        hybrid_integrals_train, data_integrals_train = [], []
        for _, group in train_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_train.append(hybrid_integral)
            data_integrals_train.append(data_integral)

        # Calculate MAPE and RRMSE for training
        mape_train = calculate_mape(np.array(data_integrals_train), np.array(hybrid_integrals_train))
        rrmse_train = calculate_rrmse(np.array(data_integrals_train), np.array(hybrid_integrals_train))

        # Similarly for testing
        hybrid_integrals_test, data_integrals_test = [], []
        for _, group in test_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_test.append(hybrid_integral)
            data_integrals_test.append(data_integral)

        mape_test = calculate_mape(np.array(data_integrals_test), np.array(hybrid_integrals_test))
        rrmse_test = calculate_rrmse(np.array(data_integrals_test), np.array(hybrid_integrals_test))

        # Calculate RMSE
        rmse = calculate_rmse(
            (y_test + test_data['Power_phys']),
            (test_data['y_pred'] + test_data['Power_phys'])
        )

        # Append results
        results.append((fold_num, rmse, rrmse_train, mape_train, rrmse_test, mape_test))
        models.append(model)

        # Print fold results
        print(f"Vehicle: {selected_car}, Fold: {fold_num}")
        print(f"Train - MAPE: {mape_train:.2f}, RRMSE: {rrmse_train:.2f}")
        print(f"Test - MAPE: {mape_test:.2f}, RRMSE: {rrmse_test:.2f}")

    # Select the best model based on median MAPE
    median_mape = np.median([result[5] for result in results])
    median_index = np.argmin([abs(result[3] - median_mape) for result in results])
    best_model = models[median_index]

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save the best model
        if best_model:
            model_file = os.path.join(save_dir, f"{model_name}_best_model_{selected_car}.json")
            best_model.save_model(model_file)
            print(f"Best model for {selected_car} saved with MAPE: {median_mape}")

        # Save the scaler
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler_{selected_car}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at {scaler_path}")

    return results, scaler, best_lambda

def process_file_with_trained_model(file, model, scaler):
    """
    Applies the trained model to a single file to add the 'Power_hybrid' column.
    """
    try:
        data = pd.read_csv(file)
        required_columns = ['speed', 'acceleration', 'ext_temp', 'Power_phys']
        if all(col in data.columns for col in required_columns):
            # Convert 'time' column to datetime
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

            # Calculate rolling statistics with a window of 5
            data['mean_accel_10'] = data['acceleration'].rolling(window=5).mean().bfill()
            data['std_accel_10'] = data['acceleration'].rolling(window=5).std().bfill()
            data['mean_speed_10'] = data['speed'].rolling(window=5).mean().bfill()
            data['std_speed_10'] = data['speed'].rolling(window=5).std().bfill()

            feature_cols = ['speed', 'acceleration', 'ext_temp', 'mean_accel_10',
                           'std_accel_10', 'mean_speed_10', 'std_speed_10', 'slope']
            # Check if all feature columns are present
            if not all(col in data.columns for col in feature_cols):
                print(f"File {file} is missing some feature columns.")
                return

            # Use the provided scaler to scale all necessary features
            features = data[feature_cols]
            features_scaled = scaler.transform(features)

            # Predict the residual using the trained model
            predicted_residual = model.predict(features_scaled)

            # Calculate the hybrid power
            data['Power_hybrid'] = predicted_residual + data['Power_phys']

            # Define columns to save
            if 'altitude' in data.columns:
                save_columns = ['time', 'speed', 'acceleration', 'ext_temp', 'mean_accel_10',
                                'std_accel_10', 'mean_speed_10', 'std_speed_10', 'slope',
                                'int_temp', 'soc', 'soh', 'chrg_cable_conn', 'pack_volt',
                                'pack_current', 'altitude', 'Power_data', 'Power_phys',
                                'Power_hybrid', 'Power_ml']
            else:
                save_columns = ['time', 'speed', 'acceleration', 'ext_temp', 'mean_accel_10',
                                'std_accel_10', 'mean_speed_10', 'std_speed_10', 'slope',
                                'int_temp', 'soc', 'soh', 'chrg_cable_conn', 'pack_volt',
                                'pack_current', 'Power_data', 'Power_phys',
                                'Power_hybrid', 'Power_ml']

            # Save the updated file
            data.to_csv(file, columns=save_columns, index=False)

            print(f"Processed file {file}")
        else:
            print(f"File {file} does not contain required columns: {required_columns}")
    except Exception as e:
        print(f"Error processing file {file}: {e}")

def add_predicted_power_column(files, model_path, scaler):
    """
    Applies the trained model to multiple files to add the 'Power_hybrid' column.
    """
    try:
        # Load the trained model
        model = xgb.XGBRegressor()
        model.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_with_trained_model, file, model, scaler) for file in files]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing file: {e}")