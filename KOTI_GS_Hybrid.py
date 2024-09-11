def process_file_with_trained_model(file, model, scaler):
    try:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns and 'Power_phys' in data.columns:
            # Calculate absolute acceleration
            data['abs_acceleration'] = data['acceleration'].abs()

            # 'time' 열을 datetime 형식으로 변환
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

            # 첫 번째 시간으로부터의 초 차이를 계산한 'elapsed_time' 열 추가
            data['elapsed_time'] = (data['time'] - data['time'].iloc[0]).dt.total_seconds()

            # 이동 평균 및 표준편차 계산
            data['mean_accel_10'] = data['acceleration'].rolling(window=5).mean()
            data['std_accel_10'] = data['acceleration'].rolling(window=5).std()
            data['mean_abs_accel_10'] = data['abs_acceleration'].rolling(window=5).mean()
            data['std_abs_accel_10'] = data['abs_acceleration'].rolling(window=5).std()
            data['mean_speed_10'] = data['speed'].rolling(window=5).mean()
            data['std_speed_10'] = data['speed'].rolling(window=5).std()
            data['mean_accel_40'] = data['acceleration'].rolling(window=20).mean()
            data['std_accel_40'] = data['acceleration'].rolling(window=20).std()
            data['mean_abs_accel_40'] = data['abs_acceleration'].rolling(window=20).mean()
            data['std_abs_accel_40'] = data['abs_acceleration'].rolling(window=20).std()
            data['mean_speed_40'] = data['speed'].rolling(window=20).mean()
            data['std_speed_40'] = data['speed'].rolling(window=20).std()

            # Forward fill to replace NaNs with the first available value
            data[['mean_accel_10','std_accel_10', 'mean_abs_accel_10', 'std_abs_accel_10', 'mean_speed_10', 'std_speed_10', 'mean_accel_40', 'std_accel_40', 'mean_abs_accel_40', 'std_abs_accel_40', 'mean_speed_40', 'std_speed_40']] = data[['mean_accel_10','std_accel_10', 'mean_abs_accel_10', 'std_abs_accel_10', 'mean_speed_10', 'std_speed_10', 'mean_accel_40', 'std_accel_40', 'mean_abs_accel_40', 'std_abs_accel_40', 'mean_speed_40', 'std_speed_40']].ffill()

            # Use the provided scaler to scale all necessary features
            features = data[['speed', 'acceleration', 'ext_temp', 'mean_accel_10','std_accel_10', 'mean_abs_accel_10', 'std_abs_accel_10', 'mean_speed_10', 'std_speed_10', 'mean_accel_40', 'std_accel_40', 'mean_abs_accel_40', 'std_abs_accel_40', 'mean_speed_40', 'std_speed_40', 'elapsed_time']]
            features_scaled = scaler.transform(features)

            # Predict the residual using the trained model
            predicted_residual = model.predict(features_scaled)

            # Calculate the hybrid power
            data['Power_hybrid'] = predicted_residual + data['Power_phys']
            save_column = ['time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh',
                    'chrg_cable_conn', 'pack_volt', 'pack_current', 'Power_data', 'Power_phys',
                    'Power_hybrid', 'Power_ml']
            # Save the updated file
            data.to_csv(file, columns = save_column, index=False)

            print(f"Processed file {file}")
        else:
            print(f"File {file} does not contain required columns 'speed', 'acceleration', or 'Power_phys'.")
    except Exception as e:
        print(f"Error processing file {file}: {e}")

def add_predicted_power_column(files, model_path, scaler):
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