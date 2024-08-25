import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

def plot_power(file_lists, target):
    for file in file_lists:
        data = pd.read_csv(file)
        parts = file.split(os.sep)
        file_name = parts[-1]
        name_parts = file_name.split('_')
        trip_info = (name_parts[2] if 'altitude' in name_parts else name_parts[1]).split('.')[0]

        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60

        power_data = np.array(data['Power_data']) / 1000
        power_phys = np.array(data['Power_phys']) / 1000
        power_hybrid = np.array(data['Power_hybrid']) / 1000

        if target == 'comparison':
            plt.figure(figsize=(10, 6))
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Power and Physics Model Power (kW)')
            plt.plot(t_min, power_data, label='Data Power (kW)', color='tab:blue', alpha=0.6)
            plt.plot(t_min, power_phys, label='Physics Model Power (kW)', color='tab:red', alpha=0.6)
            plt.ylim([-100, 100])

            plt.legend(loc='upper left')
            plt.title('Data Power vs. Physics Model Power')
            plt.tight_layout()
            plt.show()

        elif target == 'hybrid':
            plt.figure(figsize=(10, 6))
            plt.xlabel('Time (minutes)')
            plt.ylabel('Data Power and Hybrid Model Power (kW)')
            plt.plot(t_min, power_data, label='Data Power (kW)', color='tab:blue', alpha=0.6)
            plt.plot(t_min, power_hybrid, label='Hybrid Model Power (kW)', color='tab:green', alpha=0.6)
            plt.ylim([-100, 100])

            plt.legend(loc='upper left')
            plt.title('Data Power vs. Hybrid Model Power')
            plt.tight_layout()
            plt.show()

def process_files(files, scaler=None):
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6
    ACCELERATION_MIN = -15
    ACCELERATION_MAX = 9
    TEMP_MIN = -30
    TEMP_MAX = 50

    df_list = []

    for file in files:
            data = pd.read_csv(file)
            data['Residual'] = data['Power_data'] - data['Power_phys']
            processed_data = data[
                    ['time', 'speed', 'acceleration', 'ext_temp', 'Residual', 'Power_phys', 'Power_data']]
            df_list.append(processed_data)

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    full_data = pd.concat(df_list, ignore_index=True)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame([[SPEED_MIN, ACCELERATION_MIN, TEMP_MIN], [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX]],
                                columns=['speed', 'acceleration', 'ext_temp']))

    full_data[['speed', 'acceleration', 'ext_temp']] = scaler.transform(
        full_data[['speed', 'acceleration', 'ext_temp']])

    return full_data, scaler
def cross_validate(file_lists, selected_car, boost_round, save_dir=None):
    model_name = "XGB"
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []
    best_model = None

    files = file_lists

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
            'device': 'cpu',
            'eval_metric': ['rmse'],
            'lambda': 10000
        }

        evals = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(params, dtrain, num_boost_round=boost_round, evals=evals)
        y_pred = model.predict(dtest)

        rmse = np.sqrt(np.mean(np.square((y_test + test_data['Power_phys']) - (y_pred + test_data['Power_phys']))))
        rrmse = rmse / np.mean(y_test + test_data['Power_phys'])
        results.append((fold_num, rrmse, rmse))
        models.append(model)
        print(f"Vehicle: {selected_car}, Fold: {fold_num}, Boost Rounds: {boost_round}, RMSE : {rmse}, RRMSE: {rrmse}")

    median_rrmse = np.median([result[1] for result in results])
    median_index = np.argmin([abs(result[1] - median_rrmse) for result in results])
    best_model = models[median_index]

    if save_dir:
        if not os.path.exists(os.path.join(save_dir, "models")):
            os.makedirs(os.path.join(save_dir, "models"))
        if best_model:
            model_file = os.path.join(os.path.join(save_dir, "models"), f"{model_name}_best_model_{selected_car}.json")
            best_model.save_model(model_file)
            print(f"Best model for {selected_car} with boost round {boost_round} saved with RRMSE: {median_rrmse}")

        scaler_path = os.path.join(os.path.join(save_dir, "models"), f'{model_name}_scaler_{selected_car}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at {scaler_path}")
    print(f"{model_file}")
    return results, scaler

def add_predicted_power(files, model_path, scaler):
    # 모델 로드
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    for file in files:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns and 'Power_phys' in data.columns:
            features = data[['speed', 'acceleration', 'ext_temp']]
            features_scaled = scaler.transform(features)

            predicted_residual = model.predict(features_scaled)

            data['Power_hybrid'] = predicted_residual + data['Power_phys']

            data.to_csv(file, index=False)
            print(f"Processed file {file}")


file_lists = []
root_directory = r"D:\openlab"

for dirpath, dirnames, filenames in os.walk(root_directory):
    for file in filenames:
        if file.endswith('.csv'):
            file_path = os.path.join(dirpath, file)
            file_lists.append(file_path)

selected_car = 'EV6'

# 1단계: 'comparison' 그래프 플로팅
plot_power(file_lists[0:4], target='comparison')

# # 2단계: 머신러닝 모델 학습 및 검증
boost_rounds = 100
results, scaler = cross_validate(file_lists, selected_car, boost_rounds, root_directory)

# 3단계: 'hybrid' 컬럼 추가 및 그래프 플로팅
model_path = os.path.join(root_directory, "models", f"XGB_best_model_{selected_car}.json")
add_predicted_power(file_lists, model_path, scaler)
plot_power(file_lists[0:4], target='hybrid')


