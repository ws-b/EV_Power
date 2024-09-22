import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from GS_Functions import calculate_rrmse, calculate_rmse, calculate_mape
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns
import matplotlib.pyplot as plt
from GS_plot import plot_shap_values

def process_single_file(file):
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            # Compute Residual Ratio
            epsilon = 1e-8  # To prevent division by zero
            data['Residual_Ratio'] = data['Residual'] / (data['Power_phys'] + epsilon)
            # Convert 'time' to datetime
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

            # Calculate 'jerk' (rate of change of acceleration)
            data['jerk'] = data['acceleration'].diff().fillna(0)

            return data[['time', 'speed', 'acceleration', 'jerk', 'ext_temp', 'Residual',
                         'Residual_Ratio', 'Power_phys', 'Power_data']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None

def process_files(files, scaler=None, residual_scaler=None, kmeans=None, n_clusters=5):
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6  # Convert 230km/h to m/s
    ACCELERATION_MIN = -15  # m/s^2
    ACCELERATION_MAX = 9    # m/s^2
    JERK_MIN = -10          # Minimum rate of change of acceleration
    JERK_MAX = 10           # Maximum rate of change of acceleration
    TEMP_MIN = -30
    TEMP_MAX = 50

    feature_cols = ['speed', 'acceleration', 'jerk', 'ext_temp',
                    'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10']

    df_list = []
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                data = future.result()
                if data is not None:
                    # Add Trip ID for each data
                    data['trip_id'] = files.index(file)

                    # Calculate rolling statistics
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
            [SPEED_MIN, ACCELERATION_MIN, JERK_MIN, TEMP_MIN, 0, 0, 0, 0],
            [SPEED_MAX, ACCELERATION_MAX, JERK_MAX, TEMP_MAX, 1, 1, 1, 1]
        ], columns=feature_cols))

    # Apply scaling to all features
    full_data[feature_cols] = scaler.transform(full_data[feature_cols])

    # Residual Ratio scaling
    if residual_scaler is None:
        residual_scaler = StandardScaler()
        residual_scaler.fit(full_data[['Residual_Ratio']])
    full_data['Residual_Ratio_scaled'] = residual_scaler.transform(full_data[['Residual_Ratio']])

    # Clustering features
    clustering_features = ['speed', 'acceleration', 'jerk', 'Residual_Ratio_scaled']

    # Apply K-Means clustering
    if kmeans is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(full_data[clustering_features])

    # Add cluster labels
    full_data['cluster_label'] = kmeans.predict(full_data[clustering_features])

    return full_data, scaler, residual_scaler, kmeans

def integrate_and_compare(trip_data):
    # Sort by 'time'
    trip_data = trip_data.sort_values(by='time')

    # Convert 'time' to seconds
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # Integrate 'Power_phys + y_pred' using trapezoidal rule
    trip_data['Power_hybrid'] = trip_data['Power_phys'] + trip_data['y_pred']
    hybrid_integral = np.trapz(trip_data['Power_hybrid'].values, time_seconds)

    # Integrate 'Power_data' using trapezoidal rule
    data_integral = np.trapz(trip_data['Power_data'].values, time_seconds)

    # Return integrated values
    return hybrid_integral, data_integral

def cross_validate(vehicle_files, selected_car, precomputed_lambda=None, plot=None, save_dir="models", n_clusters=5):
    model_name = "XGB"

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []
    best_model = None
    scaler = None
    residual_scaler = None
    kmeans = None

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]

    for fold_num, (train_index, test_index) in enumerate(kf.split(files), 1):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        # Process train and test sets
        train_data, scaler, residual_scaler, kmeans = process_files(train_files, scaler, residual_scaler, kmeans, n_clusters)
        test_data, _, _, _ = process_files(test_files, scaler, residual_scaler, kmeans, n_clusters)

        # Group by Trip ID for integration
        train_trip_groups = train_data.groupby('trip_id')
        test_trip_groups = test_data.groupby('trip_id')

        feature_cols = ['speed', 'acceleration', 'jerk', 'ext_temp',
                        'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10', 'cluster_label']
        # Prepare data for training
        X_train = train_data[feature_cols].to_numpy()
        y_train = train_data['Residual'].to_numpy()

        X_test = test_data[feature_cols].to_numpy()
        y_test = test_data['Residual'].to_numpy()

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Train the model
        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': ['rmse'],
            'lambda': precomputed_lambda if precomputed_lambda else 1.0,
            'eta': 0.3
        }

        evals = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(params, dtrain, num_boost_round=150, evals=evals)
        train_data['y_pred'] = model.predict(dtrain)
        test_data['y_pred'] = model.predict(dtest)

        if plot:
            # Calculate and plot SHAP values
            plot_shap_values(model, X_train, feature_cols)

        # Calculate integrals and metrics for Train set
        hybrid_integrals_train, data_integrals_train = [], []
        for _, group in train_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_train.append(hybrid_integral)
            data_integrals_train.append(data_integral)

        # Calculate MAPE and RRMSE
        mape_train = calculate_mape(np.array(data_integrals_train), np.array(hybrid_integrals_train))
        rrmse_train = calculate_rrmse(np.array(data_integrals_train), np.array(hybrid_integrals_train))

        # Calculate integrals and metrics for Test set
        hybrid_integrals_test, data_integrals_test = [], []
        for _, group in test_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_test.append(hybrid_integral)
            data_integrals_test.append(data_integral)

        # Calculate MAPE and RRMSE
        mape_test = calculate_mape(np.array(data_integrals_test), np.array(hybrid_integrals_test))
        rrmse_test = calculate_rrmse(np.array(data_integrals_test), np.array(hybrid_integrals_test))

        rmse = calculate_rmse((y_test + test_data['Power_phys']), (test_data['y_pred'] + test_data['Power_phys']))

        results.append((fold_num, rmse, rrmse_train, mape_train, rrmse_test, mape_test))
        models.append(model)

        # Print results for each fold
        print(f"Vehicle: {selected_car}, Fold: {fold_num}, n_clusters: {n_clusters}")
        print(f"Train - MAPE: {mape_train:.2f}, RRMSE: {rrmse_train:.2f}")
        print(f"Test - MAPE: {mape_test:.2f}, RRMSE: {rrmse_test:.2f}")

        # ===================== Cluster Analysis and Visualization ===================== #

        # (a) Cluster-wise statistical summary
        cluster_summary = train_data.groupby('cluster_label').agg({
            'Residual_Ratio': ['mean', 'median', 'std', 'min', 'max'],
            'speed': ['mean', 'median', 'std', 'min', 'max'],
            'acceleration': ['mean', 'median', 'std', 'min', 'max'],
            'jerk': ['mean', 'median', 'std', 'min', 'max']
        }).reset_index()
        print(f"\nCluster Summary for Fold {fold_num}:\n", cluster_summary)

        # Define a consistent color palette based on the number of clusters
        num_clusters = train_data['cluster_label'].nunique()
        palette = sns.color_palette('viridis', num_clusters)

        if plot:
            # (c) Visualize combinations of features by cluster
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='speed', y='Residual_Ratio', hue='cluster_label', data=train_data, palette=palette)
            plt.title(f'Fold {fold_num} - Residual Ratio vs. Speed by Cluster')
            plt.xlabel('Speed')
            plt.ylabel('Residual Ratio')
            plt.legend(title='Cluster Label')
            plt.show()

            # Analyze speed and acceleration
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='acceleration', y='Residual_Ratio', hue='cluster_label', data=train_data, palette=palette)
            plt.title(f'Fold {fold_num} - Residual Ratio vs. Acceleration by Cluster')
            plt.xlabel('Acceleration')
            plt.ylabel('Residual Ratio')
            plt.legend(title='Cluster Label')
            plt.show()

        # (d) Label driving situations based on Residual Ratio and acceleration
        residual_ratio_threshold_high = train_data['Residual_Ratio'].quantile(0.75)
        residual_ratio_threshold_low = train_data['Residual_Ratio'].quantile(0.25)
        acceleration_threshold = 0.1  # Adjust based on data distribution

        def infer_driving_situation(row):
            residual_ratio = row['Residual_Ratio']
            acceleration = row['acceleration']

            if residual_ratio > residual_ratio_threshold_high:
                if acceleration > acceleration_threshold:
                    return 'uphill_acceleration'
                elif acceleration < -acceleration_threshold:
                    return 'uphill_deceleration'
                else:
                    return 'uphill_cruise'
            elif residual_ratio_threshold_low < residual_ratio < residual_ratio_threshold_high:
                if acceleration > acceleration_threshold:
                    return 'flatroad_acceleration'
                elif acceleration < -acceleration_threshold:
                    return 'flatroad_deceleration'
                else:
                    return 'flatroad_cruise'

            elif residual_ratio < residual_ratio_threshold_low:
                if acceleration > acceleration_threshold:
                    return 'downhill_acceleration'
                elif acceleration < -acceleration_threshold:
                    return 'downhill_deceleration'
                else:
                    return 'downhill_cruise'
            else:
                return 'undefined'

        train_data['driving_situation'] = train_data.apply(infer_driving_situation, axis=1)

        # Print distribution of driving situations by cluster
        cluster_driving_situations = train_data.groupby('cluster_label')['driving_situation'].value_counts(normalize=True).unstack().fillna(0)
        print(f"\nCluster Driving Situations for Fold {fold_num}:\n", cluster_driving_situations)

        if plot:
            # Visualize distribution of driving situations by cluster
            cluster_driving_situations.plot(kind='bar', stacked=True, figsize=(10, 7), colormap='viridis')
            plt.title(f'Fold {fold_num} - Cluster-wise Driving Situations Distribution')
            plt.ylabel('Proportion')
            plt.xlabel('Cluster Label')
            plt.legend(title='Driving Situation', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

        # ====================================================================== #

    # Select the best model (based on median MAPE)
    median_mape = np.median([result[5] for result in results])
    median_index = np.argmin([abs(result[5] - median_mape) for result in results])
    best_model = models[median_index]

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save the best model
        if best_model:
            model_file = os.path.join(save_dir, f"{model_name}_best_model_{selected_car}_k{n_clusters}.json")
            best_model.save_model(model_file)
            print(f"Best model for {selected_car} saved with MAPE: {median_mape}")

        # Save the scaler
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler_{selected_car}_k{n_clusters}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at {scaler_path}")

        # Save the Residual Ratio scaler
        residual_scaler_path = os.path.join(save_dir, f'{model_name}_residual_ratio_scaler_{selected_car}_k{n_clusters}.pkl')
        with open(residual_scaler_path, 'wb') as f:
            pickle.dump(residual_scaler, f)
        print(f"Residual Ratio scaler saved at {residual_scaler_path}")

        # Save the K-Means model
        kmeans_path = os.path.join(save_dir, f'{model_name}_kmeans_{selected_car}_k{n_clusters}.pkl')
        with open(kmeans_path, 'wb') as f:
            pickle.dump(kmeans, f)
        print(f"K-Means model saved at {kmeans_path}")

    return results, scaler, residual_scaler, kmeans

def process_file_with_trained_model(file, model, scaler, residual_scaler, kmeans):
    try:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns and 'Power_phys' in data.columns:
            # Convert 'time' to datetime
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

            # Calculate 'jerk'
            data['jerk'] = data['acceleration'].diff().fillna(0)

            # Rolling statistics
            data['mean_accel_10'] = data['acceleration'].rolling(window=5).mean().bfill()
            data['std_accel_10'] = data['acceleration'].rolling(window=5).std().bfill()
            data['mean_speed_10'] = data['speed'].rolling(window=5).mean().bfill()
            data['std_speed_10'] = data['speed'].rolling(window=5).std().bfill()

            # Compute Residual and Residual Ratio
            data['Residual'] = data['Power_data'] - data['Power_phys']
            epsilon = 1e-8  # To prevent division by zero
            data['Residual_Ratio'] = data['Residual'] / (data['Power_phys'] + epsilon)

            # Scale features
            feature_cols = ['speed', 'acceleration', 'jerk', 'ext_temp',
                            'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10']
            features = data[feature_cols]
            features_scaled = scaler.transform(features)

            # Scale Residual Ratio
            data['Residual_Ratio_scaled'] = residual_scaler.transform(data[['Residual_Ratio']])

            # Clustering features
            clustering_features = np.hstack([features_scaled[:, :3], data['Residual_Ratio_scaled'].reshape(-1, 1)])

            # Predict cluster labels
            data['cluster_label'] = kmeans.predict(clustering_features)

            # Add cluster label to features
            features_scaled = np.hstack([features_scaled, data['cluster_label'].reshape(-1, 1)])

            # Predict the residual using the trained model
            dtest = xgb.DMatrix(features_scaled)
            predicted_residual = model.predict(dtest)

            # Calculate the hybrid power
            data['Power_hybrid'] = predicted_residual + data['Power_phys']
            save_column = ['time', 'speed', 'acceleration', 'ext_temp', 'Power_data', 'Power_phys',
                           'Power_hybrid']

            # Save the updated file
            data.to_csv(file, columns=save_column, index=False)

            print(f"Processed file {file}")
        else:
            print(f"File {file} does not contain required columns 'speed', 'acceleration', or 'Power_phys'.")
    except Exception as e:
        print(f"Error processing file {file}: {e}")

def add_predicted_power_column(files, model_path, scaler_path, residual_scaler_path, kmeans_path):
    try:
        # Load the trained model
        model = xgb.Booster()
        model.load_model(model_path)

        # Load the scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Load the residual ratio scaler
        with open(residual_scaler_path, 'rb') as f:
            residual_scaler = pickle.load(f)

        # Load the K-Means model
        with open(kmeans_path, 'rb') as f:
            kmeans = pickle.load(f)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_with_trained_model, file, model, scaler, residual_scaler, kmeans) for file in files]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing file: {e}")

