import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def load_and_split_data(base_dir, vehicle_dict, train_count=400, test_count=100):
    all_files = []

    # 모든 파일 수집
    for vehicle, ids in vehicle_dict.items():
        for vid in ids:
            patterns = [
                os.path.join(base_dir, f"**/bms_{vid}-*"),
                os.path.join(base_dir, f"**/bms_altitude_{vid}-*")
            ]
            for pattern in patterns:
                all_files += glob.glob(pattern, recursive=True)

    # 전체 파일에서 랜덤하게 샘플링
    random.shuffle(all_files)
    train_files = all_files[:train_count]
    test_files = all_files[train_count:train_count + test_count]

    return train_files, test_files
"""
def load_and_split_data(base_dir, vehicle_dict, test_ratio = 0.2):
    train_files = []
    test_files = []

    for vehicle, ids in vehicle_dict.items():
        for vid in ids:
            patterns = [
                os.path.join(base_dir, f"**/bms_{vid}-*"),
                os.path.join(base_dir, f"**/bms_altitude_{vid}-*")
            ]
            files = []
            for pattern in patterns:
                files += glob.glob(pattern, recursive=True)

            # 파일을 무작위로 섞기
            random.shuffle(files)
            # 80/20으로 나누기
            split_index = int((1 - test_ratio) * len(files))
            train_files += files[:split_index]
            test_files += files[split_index:]
    return train_files, test_files
"""

def process_files(files):
    df_list = []
    for file in files:
        data = pd.read_csv(file)
        if 'Power' in data.columns and 'Power_IV' in data.columns:
            data['Residual'] = data['Power_IV'] - data['Power']
            df_list.append(data[['speed', 'acceleration', 'Residual']])

    full_data = pd.concat(df_list, ignore_index=True)
    scaler = MinMaxScaler()
    full_data[['speed', 'acceleration']] = scaler.fit_transform(full_data[['speed', 'acceleration']])
    return full_data

def plot_3d(X, y_true, y_pred):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 실제 값
    scatter = ax.scatter(X[:, 0], X[:, 1], y_true, c='blue', label='Actual Residual', alpha=0.6, edgecolors='w', s=50)

    # 예측 값
    scatter = ax.scatter(X[:, 0], X[:, 1], y_pred, c='red', label='Predicted Residual', alpha=0.6, edgecolors='w', s=50)

    # 라벨과 제목 설정
    ax.set_xlabel('Speed')
    ax.set_ylabel('Acceleration')
    ax.set_zlabel('Residual')
    ax.set_title('3D Plot of Actual vs. Predicted Residuals')
    ax.legend()

    plt.show()

def main():
    base_dir = os.path.normpath(r'D:\SamsungSTF\Processed_Data\TripByTrip')

    vehicle_dict = {
        'NiroEV': ['01241228149', '01241228151', '01241228153', '01241228154', '01241228155'],
        'Ionic5': [
            '01241227999', '01241228003', '01241228005', '01241228007', '01241228009', '01241228014',
            '01241228016', '01241228020', '01241228024', '01241228025', '01241228026', '01241228030',
            '01241228037', '01241228044', '01241228046', '01241228047', '01241248780', '01241248782',
            '01241248790', '01241248811', '01241248815', '01241248817', '01241248820', '01241248827',
            '01241364543', '01241364560', '01241364570', '01241364581', '01241592867', '01241592868',
            '01241592878', '01241592896', '01241592907', '01241597801', '01241597802', '01241248919',
            '01241321944'
        ],
        'Ionic6' : ['01241248713', '01241592904', '01241597763', '01241597804'],
        'KonaEV' : [
        '01241228102', '01241228122', '01241228123', '01241228156', '01241228197', '01241228203', '01241228204',
        '01241248726', '01241248727', '01241364621', '01241124056'
        ],
        'EV6' : [
        '01241225206', '01241228048', '01241228049', '01241228050', '01241228051', '01241228053', '01241228054',
        '01241228055', '01241228057', '01241228059', '01241228073', '01241228075', '01241228076', '01241228082',
        '01241228084', '01241228085', '01241228086', '01241228087', '01241228090', '01241228091', '01241228092',
        '01241228094', '01241228095', '01241228097', '01241228098', '01241228099', '01241228103', '01241228104',
        '01241228106', '01241228107', '01241228114', '01241228124', '01241228132', '01241228134', '01241248679',
        '01241248818', '01241248831', '01241248833', '01241248842', '01241248843', '01241248850', '01241248860',
        '01241248876', '01241248877', '01241248882', '01241248891', '01241248892', '01241248900', '01241248903',
        '01241248908', '01241248912', '01241248913', '01241248921', '01241248924', '01241248926', '01241248927',
        '01241248929', '01241248932', '01241248933', '01241248934', '01241321943', '01241321947', '01241364554',
        '01241364575', '01241364592', '01241364627', '01241364638', '01241364714'
        ],
        'GV60' : ['01241228108', '01241228130', '01241228131', '01241228136', '01241228137', '01241228138']
    }

    train_files, test_files = load_and_split_data(base_dir, vehicle_dict)
    train_data = process_files(train_files)
    test_data = process_files(test_files)

    X_train = train_data[['speed', 'acceleration']]
    y_train = train_data['Residual']
    X_test = test_data[['speed', 'acceleration']]
    y_test = test_data['Residual']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE for Residual Prediction: {rmse}")

    plot_3d(X_test.to_numpy(), y_test, y_pred)

if __name__ == "__main__":
    main()