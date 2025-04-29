# GS_Train_DNN.py
import os
import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.integrate import cumulative_trapezoid
from GS_Functions import calculate_mape # Assuming GS_Functions.py exists
from GS_plot import plot_composite_contour # Assuming GS_plot.py exists and function is adaptable
from concurrent.futures import ProcessPoolExecutor, as_completed # Keep if process_single_file uses heavy CPU
from optuna.trial import TrialState
import time

# ----------------------------
# 전역 변수 / 상수 정의
# ----------------------------
# 기존 XGBoost 파일과 동일한 상수 사용
SPEED_MIN = 0 / 3.6
SPEED_MAX = 230 / 3.6
ACCELERATION_MIN = -15
ACCELERATION_MAX = 9
TEMP_MIN = -30
TEMP_MAX = 50
ACCEL_STD_MAX = 10
SPEED_STD_MAX = 30

window_sizes = [5] # 기존 XGBoost 파일과 동일하게 유지

def generate_feature_columns():
    feature_cols = ['speed', 'acceleration', 'ext_temp']
    for w in window_sizes:
        time_window = w * 2
        feature_cols.extend([
            f'mean_accel_{time_window}',
            f'std_accel_{time_window}',
            f'mean_speed_{time_window}',
            f'std_speed_{time_window}'
        ])
    return feature_cols

FEATURE_COLS = generate_feature_columns()

# CUDA 사용 가능 여부 확인 및 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----------------------------
# 데이터 처리 함수 (XGBoost 코드와 거의 동일)
# ----------------------------
def process_single_file(file, trip_id):
    """
    파일 하나를 읽어 해당 파일이 하나의 trip으로 가정하고,
    rolling feature를 계산한 뒤 데이터를 반환.
    """
    try:
        data = pd.read_csv(file)
        # Power_phys와 Power_data가 있는지 확인 (Residual 계산에 필요)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

            # rolling feature 계산
            for w in window_sizes:
                time_window = w * 2
                # Use bfill() to handle NaNs at the beginning, similar to XGBoost version
                data[f'mean_accel_{time_window}'] = data['acceleration'].rolling(window=w, min_periods=1).mean().bfill()
                data[f'std_accel_{time_window}'] = data['acceleration'].rolling(window=w, min_periods=1).std().bfill().fillna(0)
                data[f'mean_speed_{time_window}'] = data['speed'].rolling(window=w, min_periods=1).mean().bfill()
                data[f'std_speed_{time_window}'] = data['speed'].rolling(window=w, min_periods=1).std().bfill().fillna(0)

            data['trip_id'] = trip_id
            # Ensure required columns exist before returning
            required_for_dnn = FEATURE_COLS + ['Residual', 'trip_id', 'time', 'Power_phys', 'Power_data']
            if all(col in data.columns for col in required_for_dnn):
                 # Keep only necessary columns to potentially save memory
                 return data[required_for_dnn]
            else:
                 print(f"Warning: Missing required columns in file {file}. Skipping.")
                 return None
        else:
            # print(f"Warning: 'Power_phys' or 'Power_data' missing in {file}. Skipping.")
            return None
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None

def scale_data(df, scaler=None):
    """
    FEATURE_COLS에 대해 MinMaxScaling. XGBoost 코드와 동일 로직.
    """
    if scaler is None:
        min_vals = [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN]
        max_vals = [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX]
        window_val_min = [ACCELERATION_MIN, 0, SPEED_MIN, 0]
        window_val_max = [ACCELERATION_MAX, ACCEL_STD_MAX, SPEED_MAX, SPEED_STD_MAX]

        for w in window_sizes:
            min_vals.extend(window_val_min)
            max_vals.extend(window_val_max)

        # Create a dummy DataFrame matching FEATURE_COLS order
        dummy_df_min = pd.DataFrame([min_vals], columns=FEATURE_COLS)
        dummy_df_max = pd.DataFrame([max_vals], columns=FEATURE_COLS)
        dummy_df = pd.concat([dummy_df_min, dummy_df_max], ignore_index=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit the scaler using the dummy DataFrame with correct columns
        scaler.fit(dummy_df)

    # Make sure DataFrame has columns in the correct order before transforming
    df_to_scale = df[FEATURE_COLS].copy()
    scaled_values = scaler.transform(df_to_scale)
    df[FEATURE_COLS] = scaled_values # Assign back to the original DataFrame slice
    return df, scaler


def integrate_and_compare(trip_data):
    """
    주어진 trip 데이터에 대해 예측된 Residual을 사용하여 hybrid power의 누적 에너지를 계산하고,
    실제 데이터의 누적 에너지와 비교하기 위해 반환.
    (XGBoost 코드와 동일 로직, 단 'y_pred' 대신 'dnn_pred' 사용 가정)
    """
    trip_data = trip_data.sort_values(by='time')
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # DNN 예측값을 Residual 예측으로 사용
    trip_data['Power_hybrid'] = trip_data['Power_phys'] + trip_data['dnn_pred']

    hybrid_cum_integral = cumulative_trapezoid(trip_data['Power_hybrid'].values, time_seconds, initial=0)
    hybrid_integral = hybrid_cum_integral[-1] if len(hybrid_cum_integral) > 0 else 0

    data_cum_integral = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)
    data_integral = data_cum_integral[-1] if len(data_cum_integral) > 0 else 0

    return hybrid_integral, data_integral

# ----------------------------
# PyTorch Dataset 및 모델 정의
# ----------------------------
class VehicleDataset(Dataset):
    def __init__(self, features, labels):
        # Ensure features and labels are numpy arrays before converting to tensors
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(labels, pd.Series):
            labels = labels.values

        self.features = torch.tensor(features, dtype=torch.float32)
        # Ensure labels are reshaped to [n_samples, 1] for MSELoss
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super(DNNModel, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1)) # Output layer for regression
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ----------------------------
# 학습 및 평가 루프
# ----------------------------
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for features, labels in dataloader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    # Concatenate predictions and labels from all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    return avg_loss, rmse

# ----------------------------
# Optuna Objective 함수 (DNN 용)
# ----------------------------
def dnn_cv_objective(trial, train_files, scaler):
    # 하이퍼파라미터 샘플링
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    n_layers = trial.suggest_int('n_layers', 1, 5)
    hidden_dims = [trial.suggest_int(f'n_units_l{i}', 32, 512) for i in range(n_layers)]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    epochs = 100 # Fixed epochs for tuning, use early stopping
    patience = 10 # Early stopping patience

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_val_rmse_list = []
    train_files = np.array(train_files) # KFold expects numpy array or list

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(train_files)):
        print(f"--- Optuna Fold {fold_i+1}/5 ---")
        fold_train_files = train_files[train_idx]
        fold_val_files = train_files[val_idx]

        # Process files for this fold
        fold_train_data_list = []
        with ProcessPoolExecutor() as executor: # Parallel file processing
            futures = [executor.submit(process_single_file, f, trip_id=i) for i, f in enumerate(fold_train_files)]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    fold_train_data_list.append(result)

        fold_val_data_list = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_single_file, f, trip_id=1000 + j) for j, f in enumerate(fold_val_files)]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    fold_val_data_list.append(result)

        if not fold_train_data_list or not fold_val_data_list:
             print(f"Skipping fold {fold_i+1} due to lack of data after processing.")
             # Report a large RMSE or handle appropriately for Optuna
             # Returning infinity might cause issues; maybe a very large number?
             # Or, if this happens frequently, the data processing needs review.
             return float('inf') # Or some large number

        fold_train_data = pd.concat(fold_train_data_list, ignore_index=True)
        fold_val_data = pd.concat(fold_val_data_list, ignore_index=True)


        # Scale data using the pre-fitted scaler
        fold_train_data_scaled, _ = scale_data(fold_train_data.copy(), scaler)
        fold_val_data_scaled, _ = scale_data(fold_val_data.copy(), scaler)

        # Prepare Datasets and DataLoaders
        train_dataset = VehicleDataset(fold_train_data_scaled[FEATURE_COLS], fold_train_data_scaled['Residual'])
        val_dataset = VehicleDataset(fold_val_data_scaled[FEATURE_COLS], fold_val_data_scaled['Residual'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Model, Criterion, Optimizer
        model = DNNModel(len(FEATURE_COLS), hidden_dims, dropout_rate).to(DEVICE)
        criterion = nn.MSELoss()
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else: # AdamW
            optimizer = optim.AdamW(model.parameters(), lr=lr)

        # Training loop with early stopping
        best_val_rmse = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_rmse = evaluate_model(model, val_loader, criterion)
            print(f'Fold {fold_i+1}, Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}')

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

             # Optuna Pruning (optional but recommended)
            trial.report(best_val_rmse, epoch)
            if trial.should_prune():
                print("Trial pruned by Optuna.")
                raise optuna.exceptions.TrialPruned()


        fold_val_rmse_list.append(best_val_rmse)

    # Average validation RMSE across folds
    mean_cv_rmse = np.mean(fold_val_rmse_list)
    print(f"Trial {trial.number} finished. Mean CV RMSE: {mean_cv_rmse:.4f}")
    return mean_cv_rmse

def tune_dnn_hyperparameters(train_files, scaler, selected_car, plot, n_trials=100):
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    # Wrap the objective to pass static arguments (train_files, scaler)
    objective_func = lambda trial: dnn_cv_objective(trial, train_files, scaler)
    study.optimize(objective_func, n_trials=n_trials)

    print(f"Best trial for {selected_car}: {study.best_trial.params}")
    print(f"Best CV RMSE: {study.best_value}")

    if plot:
         # --- Plotting logic (similar to XGBoost version) ---
        trials_df = study.trials_dataframe()
        trials_save_path = r"C:\Users\BSL\Desktop\Results" # Adjust path if needed
        if not os.path.exists(trials_save_path):
             os.makedirs(trials_save_path)
        trials_save = os.path.join(trials_save_path, f"{selected_car}_dnn_optuna_trials_results.csv")
        trials_df.to_csv(trials_save, index=False)
        print(f"DNN Optuna trial results saved to {trials_save}")

        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not complete_trials:
             print("No complete trials to plot.")
             return study.best_trial.params # Return params even if plotting fails

        trial_numbers = [t.number for t in complete_trials]
        trial_values = [t.value for t in complete_trials] # best_val_rmse

        plt.figure(figsize=(12, 7))
        plt.plot(trial_numbers, trial_values, marker='o', linestyle='-', label='Trials', alpha=0.7)

        best_trial = study.best_trial
        best_trial_number = best_trial.number
        best_trial_value = best_trial.value

        plt.plot(best_trial_number, best_trial_value, marker='o', markersize=10, color='red', linestyle='', label=f'Best Trial ({best_trial_value:.4f})')

        plt.xlabel('Trial Number')
        plt.ylabel('Mean CV RMSE (Lower is Better)')
        plt.title(f'DNN Optuna Hyperparameter Optimization for {selected_car}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Annotate best parameters (optional, can get crowded)
        # best_params_str = '\n'.join([f"{k}: {v}" for k, v in best_trial.params.items()])
        # plt.text(0.95, 0.95, best_params_str, transform=plt.gca().transAxes, fontsize=9, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.tight_layout()

        save_directory = r"C:\Users\BSL\Desktop\Figures\Supplementary" # Adjust path
        if not os.path.exists(save_directory):
             os.makedirs(save_directory)
        save_filename = f"FigureS10_DNN_{selected_car}.png" # Modified filename
        save_path = os.path.join(save_directory, save_filename)

        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"DNN Optuna plot saved: {save_path}")
        except Exception as e:
            print(f"Error saving Optuna plot: {e}")

        plt.show()
        # --- End Plotting ---

    return study.best_trial.params


def train_final_dnn_model(X_train, y_train, best_params):
    """ Trains the final DNN model on the entire training set """
    print("Training final DNN model...")
    input_dim = X_train.shape[1]
    hidden_dims = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]
    dropout_rate = best_params['dropout_rate']
    lr = best_params['lr']
    optimizer_name = best_params['optimizer']
    batch_size = best_params['batch_size']
    epochs = 150 # Increased epochs for final training, or adjust based on Optuna results

    train_dataset = VehicleDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = DNNModel(input_dim, hidden_dims, dropout_rate).to(DEVICE)
    criterion = nn.MSELoss()
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else: # AdamW
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0: # Print progress every 10 epochs
             print(f'Final Training Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}')
    end_time = time.time()
    print(f"Final DNN training finished in {end_time - start_time:.2f} seconds.")
    return model


# ----------------------------
# DNN 워크플로우 함수
# ----------------------------
def run_dnn_workflow(vehicle_files, selected_car, plot=False, save_dir="models_dnn", predefined_best_params=None):
    """
    PyTorch DNN 모델 학습 및 평가 워크플로우.
    """
    print(torch.cuda.is_available())  # True면 GPU 사용 가능
    print(torch.cuda.current_device())  # 현재 사용 중인 GPU 장치 번호
    print(torch.cuda.get_device_name(0))  # 장치 이름 출력
    print(torch.cuda.memory_allocated(0))  # 사용 중인 메모리 (bytes)
    print(torch.cuda.memory_reserved(0))  # 예약한 메모리 (bytes)

    start_workflow_time = time.time()
    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for the selected vehicle: {selected_car}")
        return [], None # Return empty results and None scaler

    files = vehicle_files[selected_car]
    print(f"Starting DNN workflow for {selected_car} with {len(files)} files...")

    # 1. 파일 단위 Train/Test 분할 (8:2)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    print(f"Split: {len(train_files)} train files, {len(test_files)} test files.")

    # 2. Train 데이터 처리 및 Scaler Fit
    print("Processing training files...")
    train_data_list = []
    # Use ProcessPoolExecutor for potentially faster I/O and computation
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, f, trip_id=i) for i, f in enumerate(train_files)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                train_data_list.append(result)

    if not train_data_list:
        print(f"Error: No valid training data could be processed for {selected_car}.")
        return [], None

    train_data = pd.concat(train_data_list, ignore_index=True)
    print(f"Training data shape: {train_data.shape}")

    # Fit scaler on the combined training data
    print("Fitting scaler...")
    train_data_scaled, scaler = scale_data(train_data.copy()) # Fit and transform train data
    X_train_scaled = train_data_scaled[FEATURE_COLS]
    y_train = train_data_scaled['Residual'] # Target is 'Residual'

    # 3. Test 데이터 처리 및 스케일링
    print("Processing test files...")
    test_data_list = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, f, trip_id=1000 + j) for j, f in enumerate(test_files)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                test_data_list.append(result)

    if not test_data_list:
        print(f"Warning: No valid test data could be processed for {selected_car}. Cannot evaluate.")
        # Proceed with training but return no evaluation metrics? Or stop?
        # Let's stop here as evaluation is crucial.
        return [], scaler # Return empty results but the fitted scaler

    test_data = pd.concat(test_data_list, ignore_index=True)
    print(f"Test data shape: {test_data.shape}")

    # Scale test data using the *fitted* scaler
    test_data_scaled, _ = scale_data(test_data.copy(), scaler)
    X_test_scaled = test_data_scaled[FEATURE_COLS]
    y_test = test_data_scaled['Residual'] # Ground truth residual for test set

    # -----------------------
    # 4. Hyperparameter Tuning or Use Predefined
    # -----------------------
    if predefined_best_params is None:
        print("Starting hyperparameter tuning with Optuna...")
        tune_start_time = time.time()
        # Use a smaller number of trials for faster execution if needed, e.g., n_trials=50
        best_params = tune_dnn_hyperparameters(train_files, scaler, selected_car, plot, n_trials=50)
        tune_end_time = time.time()
        print(f"Hyperparameter tuning finished in {tune_end_time - tune_start_time:.2f} seconds.")
    else:
        best_params = predefined_best_params
        print(f"Using predefined best_params: {best_params}")

    # -----------------------
    # 5. 최종 모델 학습
    # -----------------------
    # Ensure X_train_scaled and y_train are numpy arrays for the dataset
    final_model = train_final_dnn_model(X_train_scaled.values, y_train.values, best_params)

    # -----------------------
    # 6. Test Set 평가
    # -----------------------
    print("Evaluating final model on the test set...")
    test_dataset = VehicleDataset(X_test_scaled.values, y_test.values)
    # Use a larger batch size for evaluation if memory allows
    test_loader = DataLoader(test_dataset, batch_size=best_params.get('batch_size', 256) * 2, shuffle=False, num_workers=4, pin_memory=True)

    final_model.eval()
    test_preds_list = []
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(DEVICE)
            outputs = final_model(features)
            test_preds_list.append(outputs.cpu().numpy())

    # Concatenate predictions from all test batches
    y_pred_test_dnn = np.concatenate(test_preds_list, axis=0).flatten() # Flatten to 1D array

    # Calculate Test RMSE on the 'Residual' target
    test_rmse = np.sqrt(mean_squared_error(y_test.values, y_pred_test_dnn))
    print(f"Test RMSE (on Residual): {test_rmse:.4f}")

    # Calculate Test MAPE based on integrated energy
    # Add predictions back to the *original* (unscaled) test dataframe for integration
    test_data['dnn_pred'] = y_pred_test_dnn # Add the predicted residual

    test_trip_groups = test_data.groupby('trip_id')
    hybrid_integrals_test, data_integrals_test = [], []

    integration_errors = 0
    processed_trips = 0
    for trip_id, group in test_trip_groups:
        if len(group) < 2: # Need at least 2 points to integrate
             # print(f"Skipping integration for trip {trip_id}: only {len(group)} data points.")
             continue
        try:
            hybrid_integral, data_integral = integrate_and_compare(group.copy()) # Pass a copy
            # Avoid division by zero or near-zero in MAPE calculation later
            if abs(data_integral) > 1e-6: # Check if data_integral is reasonably non-zero
                 hybrid_integrals_test.append(hybrid_integral)
                 data_integrals_test.append(data_integral)
                 processed_trips += 1
            # else:
            #      print(f"Skipping trip {trip_id} for MAPE due to near-zero data integral: {data_integral}")
        except Exception as e:
            print(f"Error during integration for trip {trip_id}: {e}")
            integration_errors += 1


    mape_test = calculate_mape(np.array(data_integrals_test), np.array(hybrid_integrals_test)) if processed_trips > 0 else float('nan')

    print(f"\nTest Set Integration Metrics ({processed_trips} trips processed, {integration_errors} errors):")
    print(f"MAPE (Energy): {mape_test:.2f}%" if not np.isnan(mape_test) else "MAPE: Not Available")

    results = [{
        'rmse': test_rmse,
        'test_mape': mape_test,
        'best_params': best_params
    }]

    # -----------------------
    # 7. Plotting (Optional)
    # -----------------------
    if plot:
        print("Generating plots...")
        # Plotting composite contour requires adapting or ensuring GS_plot.plot_composite_contour works
        # It needs X_train, y_pred_train, X_test, y_pred_test1, y_pred_test2, scaler, etc.
        # We need train predictions first
        final_model.eval()
        train_preds_list = []
        # Create DataLoader for training data for prediction
        train_pred_dataset = VehicleDataset(X_train_scaled.values, y_train.values) # Use scaled features
        train_pred_loader = DataLoader(train_pred_dataset, batch_size=best_params.get('batch_size', 256) * 2, shuffle=False, num_workers=4, pin_memory=True)
        with torch.no_grad():
             for features, _ in train_pred_loader:
                  features = features.to(DEVICE)
                  outputs = final_model(features)
                  train_preds_list.append(outputs.cpu().numpy())
        y_pred_train_dnn = np.concatenate(train_preds_list, axis=0).flatten()


        # Prepare data for plot_composite_contour
        # Ensure inputs are numpy arrays
        X_train_plot = X_train_scaled[['speed', 'acceleration']].values
        y_pred_train_plot = y_pred_train_dnn
        X_test_plot = X_test_scaled[['speed', 'acceleration']].values
        y_pred_test1_plot = y_pred_test_dnn # Predicted residual on test
        # Calculate the difference between actual and predicted residual for the third plot
        y_pred_test2_plot = (y_test.values - y_pred_test_dnn) # Error in residual prediction


        # 플랫폼에 따라 저장 경로 설정
        if platform.system() == "Windows":
            save_fig_dir = r"C:\Users\BSL\Desktop\Figures\RF_Importance"
        else:
            save_fig_dir = os.path.expanduser("~/SamsungSTF/Figures/")

        try:
             plot_composite_contour(
                  X_train=X_train_plot,
                  y_pred_train=y_pred_train_plot,
                  X_test=X_test_plot,
                  y_pred_test1=y_pred_test1_plot, # DNN Predicted Residual
                  y_pred_test2=y_pred_test2_plot, # Error (Actual Residual - Predicted Residual)
                  scaler=scaler, # Pass the fitted scaler
                  selected_car=selected_car,
                  terminology1=f'{selected_car} DNN: Train Predicted Residual',
                  terminology2=f'{selected_car} DNN: Test Predicted Residual',
                  terminology3=f'{selected_car} DNN: Test Residual Error',
                  num_grids=30,
                  min_count=10,
                  save_directory=save_fig_dir, # Adjust path
                  filename_suffix="_DNN" # Add suffix to distinguish from XGB plots
             )
             print("Composite contour plot generated.")
        except Exception as e:
             print(f"Failed to generate composite contour plot: {e}")


    # -----------------------
    # 8. 모델 및 스케일러 저장
    # -----------------------
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save Model State Dictionary
        model_file = os.path.join(save_dir, f"DNN_best_model_{selected_car}.pth")
        try:
            torch.save(final_model.state_dict(), model_file)
            print(f"Best DNN model state_dict for {selected_car} saved to {model_file}")
        except Exception as e:
            print(f"Error saving DNN model state_dict: {e}")

        # Save Scaler
        scaler_path = os.path.join(save_dir, f'DNN_scaler_{selected_car}.pkl')
        try:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved at {scaler_path}")
        except Exception as e:
            print(f"Error saving scaler: {e}")

    end_workflow_time = time.time()
    print(f"DNN workflow for {selected_car} completed in {end_workflow_time - start_workflow_time:.2f} seconds.")

    return results, scaler # Return results and the scaler object