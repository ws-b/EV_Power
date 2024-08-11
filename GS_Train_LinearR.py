import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from GS_plot import plot_3d, plot_contour
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib

# 데이터 전처리 함수
def process_single_file(file):
    try:
        data = pd.read_csv(file)
        if 'Power' in data.columns and 'Power_IV' in data.columns:
            data['Residual'] = data['Power'] - data['Power_IV']
            return data[['speed', 'acceleration', 'Residual', 'Power', 'Power_IV']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None

# 여러 파일을 처리하는 함수
def process_files(files, scaler=None):
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6 # 230km/h 를 m/s 로
    ACCELERATION_MIN = -15 # m/s^2
    ACCELERATION_MAX = 9 # m/s^2

    df_list = []
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                data = future.result()
                if data is not None:
                    df_list.append((files.index(file), data))
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    # Sort the list by the original file order
    df_list.sort(key=lambda x: x[0])
    df_list = [df for _, df in df_list]

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    full_data = pd.concat(df_list, ignore_index=True)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame([[SPEED_MIN, ACCELERATION_MIN], [SPEED_MAX, ACCELERATION_MAX]], columns=['speed', 'acceleration']))

    full_data[['speed', 'acceleration']] = scaler.transform(full_data[['speed', 'acceleration']])

    return full_data, scaler


def calculate_rrmse(y_test, y_pred):
    relative_errors = (y_pred - y_test) / np.mean(y_test)
    rrmse = np.sqrt(np.mean(relative_errors ** 2))
    return rrmse

# 교차 검증 및 모델 학습 함수
def cross_validate(vehicle_files, selected_car, save_dir="models"):
    model_name = "LR"
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []
    best_model = None

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]

    for fold_num, (train_index, test_index) in enumerate(kf.split(files), 1):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        train_data, scaler = process_files(train_files)
        test_data, _ = process_files(test_files)

        X_train = train_data[['speed', 'acceleration']].to_numpy()
        y_train = train_data['Residual'].to_numpy()

        X_test = test_data[['speed', 'acceleration']].to_numpy()
        y_test = test_data['Residual'].to_numpy()

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean(((test_data['Power'] - y_pred) - (test_data['Power'] - y_test)) ** 2))
        residual2 = y_pred - y_test
        rrmse = calculate_rrmse(test_data['Power'] - y_test, test_data['Power'] - y_pred)
        results.append((fold_num, rrmse, rmse))
        models.append(model)
        print(f"Vehicle: {selected_car}, Fold: {fold_num}, RRMSE: {rrmse}")

        # Calculate the median RRMSE
        median_rrmse = np.median([result[1] for result in results])
        # Find the index of the model corresponding to the median RRMSE
        median_index = np.argmin([abs(result[1] - median_rrmse) for result in results])
        best_model = models[median_index]
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save the best model
        if best_model:
            model_file = os.path.join(save_dir, f"{model_name}_best_model_{selected_car}.joblib")
            joblib.dump(best_model, model_file)
            print(f"Best model for {selected_car} saved with RRMSE: {median_rrmse}")

            # plot_contour(X_test, y_pred, scaler, selected_car, 'Predicted Residual[1]', num_grids=400)
            # plot_contour(X_test, residual2, scaler, selected_car, 'Residual[2]',  num_grids=400)

        # Save the scaler
        scaler_path = os.path.join(save_dir, f"{model_name}_scaler_{selected_car}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at {scaler_path}")

    return results, scaler

# 학습된 모델을 사용하여 파일 처리 함수
def process_file_with_trained_model(file, model, scaler):
    try:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns and 'Power' in data.columns:
            # Use the provided scaler
            features = data[['speed', 'acceleration']]
            features_scaled = scaler.transform(features)

            predicted_residual = model.predict(features_scaled)

            data['Predicted_Power'] = data['Power'] - predicted_residual

            # Save the updated file
            data.to_csv(file, index=False)

            print(f"Processed file {file}")
        else:
            print(f"File {file} does not contain required columns 'speed', 'acceleration', or 'Power'.")
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# 여러 파일을 학습된 모델로 처리하는 함수
def add_predicted_power_column(files, model_path, scaler):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_with_trained_model, file, model, scaler) for file in files]
        for future in as_completed(futures):
            future.result()