import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV
from concurrent.futures import ProcessPoolExecutor, as_completed

def plot_power(file_lists, selected_car, target):
    for file in file_lists:
        data = pd.read_csv(file)
        parts = file.split(os.sep)
        file_name = parts[-1]
        name_parts = file_name.split('_')
        trip_info = (name_parts[2] if 'altitude' in name_parts else name_parts[1]).split('.')[0]

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60
        date = t.iloc[0].strftime('%Y-%m-%d')

        power_data = np.array(data['Power_data']) / 1000
        power_phys = np.array(data['Power_phys']) / 1000

        if target == 'comparison':
            plt.figure(figsize=(10, 6))
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Power and Physics Model Power (kW)')
            plt.plot(t_min, power_data, label='Data Power (kW)', color='tab:blue', alpha=0.6)
            plt.plot(t_min, power_phys, label='Physics Model Power (kW)', color='tab:red', alpha=0.6)
            plt.ylim([-100, 100])

            plt.text(0.99, 0.99, date, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', color='black')
            plt.text(0.01, 0.99, f'{selected_car}: '+ trip_info, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='left', color='black')

            plt.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
            plt.title('Data Power vs. Physics Model Power')
            plt.tight_layout()
            plt.show()

def process_single_file(file):
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            return data[['time', 'speed', 'acceleration', 'ext_temp', 'Residual', 'Power_phys', 'Power_data']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None

def process_files(files, scaler=None):
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6
    ACCELERATION_MIN = -15
    ACCELERATION_MAX = 9
    TEMP_MIN = -30
    TEMP_MAX = 50

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

    df_list.sort(key=lambda x: x[0])
    df_list = [df for _, df in df_list]

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    full_data = pd.concat(df_list, ignore_index=True)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame([[SPEED_MIN, ACCELERATION_MIN, TEMP_MIN], [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX]], columns=['speed', 'acceleration', 'ext_temp']))

    full_data[['speed', 'acceleration', 'ext_temp']] = scaler.transform(full_data[['speed', 'acceleration', 'ext_temp']])

    return full_data, scaler

def cross_validate(vehicle_files, selected_car, precomputed_lambda, boost_rounds, save_dir="models"):
    model_name = "XGB"
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []
    best_model = None

    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    files = vehicle_files[selected_car]
    best_lambda = precomputed_lambda

    for boost_round in boost_rounds:
        for fold_num, (train_index, test_index) in enumerate(kf.split(files), 1):
            train_files = [files[i] for i in train_index]
            test_files = [files[i] for i in test_index]

            train_data, scaler = process_files(train_files)
            test_data, _ = process_files(test_files)

            X_train = train_data[['speed', 'acceleration', 'ext_temp']].to_numpy()
            y_train = train_data['Residual'].to_numpy()

            X_test = test_data[['speed', 'acceleration', 'ext_temp']].to_numpy()
            y_test = test_data['Residual'].to_numpy()

            dtrain = xgb.DMatrix(X_train, label=y_train, weight=X_train[:, 0])
            dtest = xgb.DMatrix(X_test, label=y_test, weight=X_test[:, 0])

            params = {
                'tree_method': 'hist',
                'device': 'cuda',
                'eval_metric': ['rmse'],
                'lambda': best_lambda
            }

            evals = [(dtrain, 'train'), (dtest, 'test')]
            model = xgb.train(params, dtrain, num_boost_round=boost_round, evals=evals)
            y_pred = model.predict(dtest)

            rmse = np.sqrt(np.mean(np.square((y_test + test_data['Power_phys']) - (y_pred + test_data['Power_phys']))))
            rrmse = rmse / np.mean(y_test + test_data['Power_phys'])
            results.append((fold_num, rrmse, rmse))
            models.append(model)
            print(f"Vehicle: {selected_car}, Fold: {fold_num}, Boost Rounds: {boost_round}, RRMSE: {rrmse}")

        median_rrmse = np.median([result[1] for result in results])
        median_index = np.argmin([abs(result[1] - median_rrmse) for result in results])
        best_model = models[median_index]

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if best_model:
                model_file = os.path.join(save_dir, f"{model_name}_best_model_{selected_car}_boost_{boost_round}.json")
                best_model.save_model(model_file)
                print(f"Best model for {selected_car} with boost round {boost_round} saved with RRMSE: {median_rrmse}")

            scaler_path = os.path.join(save_dir, f'{model_name}_scaler_{selected_car}.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved at {scaler_path}")

    return results, scaler, best_lambda

def process_file_with_trained_model(file, model, scaler):
    try:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns and 'Power_phys' in data.columns:
            features = data[['speed', 'acceleration', 'ext_temp']]
            features_scaled = scaler.transform(features)

            predicted_residual = model.predict(features_scaled)

            data['Power_hybrid'] = predicted_residual + data['Power_phys']
            data.to_csv(file, index=False)
            print(f"Processed file {file}")
        else:
            print(f"File {file} does not contain required columns 'speed', 'acceleration', or 'Power_phys'.")
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

def main():
    # 파일 리스트, 차량 정보, target 설정
    file_lists = ['path_to_file_1.csv', 'path_to_file_2.csv', ...]
    selected_car = 'YourCarModel'

    # 1단계: 'comparison' 그래프 플로팅
    plot_power(file_lists, selected_car, target='comparison')

    # 2단계: 머신러닝 모델 학습 및 검증
    vehicle_files = {selected_car: file_lists}
    precomputed_lambda = 10000
    boost_rounds = [50, 100, 150]
    results, scaler, best_lambda = cross_validate(vehicle_files, selected_car, precomputed_lambda, boost_rounds)

    # 3단계: 'hybrid' 컬럼 추가 및 그래프 플로팅
    model_path = f"models/XGB_best_model_{selected_car}_boost_100.json"  # 예시: boost_round=100에서 최적 모델 선택
    add_predicted_power_column(file_lists, model_path, scaler)
    plot_power(file_lists, selected_car, target='hybrid')

if __name__ == "__main__":
    main()
