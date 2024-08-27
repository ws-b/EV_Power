import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from GS_Functions import calculate_rrmse, calculate_rmse, calculate_mape
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from GS_plot import plot_3d, plot_contour
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    SPEED_MAX = 230 / 3.6 # 230km/h 를 m/s 로
    ACCELERATION_MIN = -15 # m/s^2
    ACCELERATION_MAX = 9 # m/s^2
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

    # Sort the list by the original file order
    df_list.sort(key=lambda x: x[0])
    df_list = [df for _, df in df_list]

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    full_data = pd.concat(df_list, ignore_index=True)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame([[SPEED_MIN, ACCELERATION_MIN, TEMP_MIN], [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX]], columns=['speed', 'acceleration','ext_temp']))

    full_data[['speed', 'acceleration', 'ext_temp']] = scaler.transform(full_data[['speed', 'acceleration', 'ext_temp']])

    return full_data, scaler

def custom_obj(preds, dtrain):
    labels = dtrain.get_label()
    speed = dtrain.get_weight()  # Use weight to store speed

    grad = preds - labels
    hess = np.ones_like(grad)

    # speed가 0인 경우 제약 조건 반영
    mask = (speed == 0)
    grad[mask] = np.maximum(0, grad[mask])

    return grad, hess

def grid_search_lambda(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=X_train[:, 0])
    lambda_values = np.logspace(-3, 7, num=11)
    param_grid = {
        'tree_method': ['hist'],
        'device': ['cuda'],
        'eval_metric': ['rmse'],
        'lambda': lambda_values
    }

    model = xgb.XGBRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5,
                               verbose=1)
    grid_search.fit(X_train, y_train)

    best_lambda = grid_search.best_params_['lambda']
    print(f"Best lambda found: {best_lambda}")

    return best_lambda

def cross_validate(vehicle_files, selected_car, precomputed_lambda, plot = None, save_dir="models"):
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
    for fold_num, (train_index, test_index) in enumerate(kf.split(files), 1):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        train_data, scaler = process_files(train_files)
        test_data, _ = process_files(test_files)

        X_train = train_data[['speed', 'acceleration', 'ext_temp']].to_numpy()
        y_train = train_data['Residual'].to_numpy()

        X_test = test_data[['speed', 'acceleration', 'ext_temp']].to_numpy()
        y_test = test_data['Residual'].to_numpy()

        if best_lambda is None:
            best_lambda = grid_search_lambda(X_train, y_train)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=X_train[:, 0])
        dtest = xgb.DMatrix(X_test, label=y_test, weight=X_test[:, 0])

        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': ['rmse'],
            'lambda': best_lambda
        }

        evals = [(dtrain, 'train'), (dtest, 'test')]
        #model = xgb.train(params, dtrain, num_boost_round=150, evals=evals, obj=custom_obj)
        model = xgb.train(params, dtrain, num_boost_round=150, evals=evals)
        y_pred = model.predict(dtest)
        #
        # test_data['y_test'] = y_test + test_data['Power_phys']
        # test_data['y_pred'] = y_pred + test_data['Power_phys']
        # test_data['time'] = pd.to_datetime(test_data['time'])
        #
        # test_data['minute'] = test_data['time'].dt.floor('min')
        # grouped = test_data.groupby('minute')
        #
        # y_test_integrated = grouped.apply(lambda group: np.trapz(group['y_test'], x=group['time'].astype('int64') / 1e9))
        # y_pred_integrated = grouped.apply(lambda group: np.trapz(group['y_pred'], x=group['time'].astype('int64') / 1e9))

        # rmse = calculate_rmse(y_test_integrated, y_pred_integrated)
        # rrmse = calculate_rrmse(y_test_integrated, y_pred_integrated)
        mape = calculate_mape(y_test + test_data['Power_phys'], y_pred + test_data['Power_phys'])
        rmse = calculate_rmse((y_test + test_data['Power_phys']), (y_pred + test_data['Power_phys']))
        rrmse = calculate_rrmse((y_test + test_data['Power_phys']), (y_pred + test_data['Power_phys']))
        residual2 = y_test - y_pred
        results.append((fold_num, rrmse, rmse, mape))
        models.append(model)
        print(f"Vehicle: {selected_car}, Fold: {fold_num}, RMSE: {rmse}, RRMSE: {rrmse}, MAPE: {mape}")

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
            model_file = os.path.join(save_dir, f"{model_name}_best_model_{selected_car}.json")
            surface_plot = os.path.join(save_dir, f"{model_name}_best_model_{selected_car}_plot.html")
            best_model.save_model(model_file)
            print(f"Best model for {selected_car} saved with RRMSE: {median_rrmse}")
            # if plot:
            #     # plot_3d(X_test, y_test, y_pred, fold_num, selected_car, scaler, 400, 30, output_file=surface_plot)
            #
            #     plot_contour(X_test, y_pred, scaler, selected_car, 'Predicted Residual[1]', num_grids=400)
            #     plot_contour(X_test, residual2, scaler, selected_car, 'Residual[2]', num_grids=400)

        # Save the scaler
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler_{selected_car}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at {scaler_path}")

    return results, scaler, best_lambda


def process_file_with_trained_model(file, model, scaler):
    try:
        data = pd.read_csv(file)
        if 'speed' in data.columns and 'acceleration' in data.columns and 'Power_phys' in data.columns:
            # Use the provided scaler
            features = data[['speed', 'acceleration', 'ext_temp']]
            features_scaled = scaler.transform(features)

            predicted_residual = model.predict(features_scaled)

            data['Power_hybrid'] = predicted_residual + data['Power_phys']

            # Save the updated file
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
