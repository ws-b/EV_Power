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
from concurrent.futures import ProcessPoolExecutor, as_completed
from optuna.trial import TrialState
import time

try:
    from GS_Functions import calculate_mape
except ImportError:
    print("Warning: GS_Functions.py not found. Using a placeholder for calculate_mape.")
    def calculate_mape(true_values, pred_values):
        true_values, pred_values = np.asarray(true_values), np.asarray(pred_values)
        mask = true_values != 0
        if np.sum(mask) == 0:
            return float('nan')
        return np.mean(np.abs((true_values[mask] - pred_values[mask]) / true_values[mask])) * 100

# ----------------------------
# 전역 변수 / 상수 정의
# ----------------------------
SPEED_MIN = 0 / 3.6
SPEED_MAX = 230 / 3.6
ACCELERATION_MIN = -15
ACCELERATION_MAX = 9
TEMP_MIN = -30
TEMP_MAX = 50

FEATURE_COLS = ['speed', 'acceleration', 'ext_temp']
NUM_FEATURES = len(FEATURE_COLS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----------------------------
# 데이터 처리 함수
# ----------------------------
def process_single_file_lstm(file, trip_id):
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns and all(c in data.columns for c in FEATURE_COLS):
            data['Residual'] = data['Power_data'] - data['Power_phys']
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            data['trip_id'] = trip_id
            data['original_index'] = data.index # 원래 DataFrame index 보존

            required_cols = FEATURE_COLS + ['Residual', 'trip_id', 'time', 'Power_phys', 'Power_data', 'original_index']
            return data[required_cols]
        else:
            missing = [c for c in FEATURE_COLS + ['Power_phys', 'Power_data'] if c not in data.columns]
            # print(f"Warning: Missing required columns {missing} in {file}. Skipping.")
            return None
    except Exception as e:
        print(f"Error processing file {file} for LSTM: {e}")
        return None

def scale_data_lstm(df_list, scaler=None):
    if scaler is None:
        combined_features = pd.concat([df[FEATURE_COLS] for df in df_list if df is not None and not df.empty], ignore_index=True)
        if combined_features.empty:
            print("Error: No data available to fit the scaler.")
            return df_list, None

        min_vals = [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN]
        max_vals = [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX]
        dummy_df_min = pd.DataFrame([min_vals], columns=FEATURE_COLS)
        dummy_df_max = pd.DataFrame([max_vals], columns=FEATURE_COLS)
        dummy_df = pd.concat([dummy_df_min, dummy_df_max], ignore_index=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(dummy_df)
        print("Scaler fitted.")

    scaled_df_list = []
    for df in df_list:
        if df is not None and not df.empty:
            df_copy = df.copy()
            features_to_scale = df_copy[FEATURE_COLS]
            scaled_features = scaler.transform(features_to_scale)
            df_copy[FEATURE_COLS] = scaled_features
            scaled_df_list.append(df_copy)
        else:
            scaled_df_list.append(None)
    return scaled_df_list, scaler

def create_sequences(data, sequence_length):
    xs, ys, idxs, trip_ids_for_seq = [], [], [], [] 
    feature_values = data[FEATURE_COLS].values
    target_values = data['Residual'].values
    original_indices = data['original_index'].values
    trip_id_value = data['trip_id'].iloc[0] 

    if len(data) >= sequence_length:
        for i in range(len(data) - sequence_length + 1):
            sequence_x = feature_values[i : i + sequence_length]
            sequence_y = target_values[i + sequence_length - 1]
            target_idx = original_indices[i + sequence_length - 1]
            
            xs.append(sequence_x)
            ys.append(sequence_y)
            idxs.append(target_idx)
            trip_ids_for_seq.append(trip_id_value) 

    return np.array(xs), np.array(ys), np.array(idxs), np.array(trip_ids_for_seq)

# ----------------------------
# PyTorch Dataset 및 LSTM 모델
# ----------------------------
class VehicleSequenceDataset(Dataset):
    def __init__(self, scaled_trip_df_list, sequence_length):
        self.sequence_length = sequence_length
        self.sequences_data = []
        self.targets_data = []
        self.target_indices_data = []
        self.target_trip_ids_data = [] 

        print(f"Creating sequences with length {sequence_length}...")
        num_processed = 0
        total_sequences = 0
        # trip_id_value를 루프 전에 정의되지 않도록 초기화 (np.empty의 dtype 결정용)
        # 실제 trip_id_value는 루프 내에서 첫번째 유효한 df로부터 얻어짐
        first_trip_id_example = None 

        for df in scaled_trip_df_list:
            if df is not None and not df.empty and len(df) >= sequence_length:
                if first_trip_id_example is None: # 첫번째 유효한 df에서 trip_id 예시 저장
                    first_trip_id_example = df['trip_id'].iloc[0]
                
                seqs, targs, idxs, t_ids = create_sequences(df, sequence_length)
                if len(seqs) > 0:
                    self.sequences_data.append(seqs)
                    self.targets_data.append(targs)
                    self.target_indices_data.append(idxs)
                    self.target_trip_ids_data.append(t_ids) 
                    num_processed += 1
                    total_sequences += len(seqs)

        if not self.sequences_data:
            print("Warning: No sequences could be created.")
            self.sequences = np.empty((0, sequence_length, NUM_FEATURES))
            self.targets = np.empty((0,))
            self.target_indices = np.empty((0,), dtype=int)
            # first_trip_id_example이 None일 수 있으므로 object로 설정
            self.target_trip_ids = np.empty((0,), dtype=object) 
        else:
            self.sequences = np.concatenate(self.sequences_data, axis=0)
            self.targets = np.concatenate(self.targets_data, axis=0)
            self.target_indices = np.concatenate(self.target_indices_data, axis=0)
            self.target_trip_ids = np.concatenate(self.target_trip_ids_data, axis=0) 
        print(f"Processed {num_processed} trips, created {total_sequences} sequences.")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if idx >= len(self.targets):
            raise IndexError("Index out of bounds")
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32).view(1)
        target_idx = self.target_indices[idx]
        target_trip_id = self.target_trip_ids[idx] 
        return sequence, target, target_idx, target_trip_id 

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ----------------------------
# 학습 및 평가 루프
# ----------------------------
def train_epoch_lstm(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for sequences, targets, _, _ in dataloader: 
        sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / len(dataloader)

def evaluate_model_lstm(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds_batches, all_labels_batches, all_indices_batches = [], [], []
    all_trip_ids_collected = [] # 변경: trip ID들을 모으는 리스트

    with torch.no_grad():
        for sequences, targets, target_idxs_batch, target_trip_ids_batch in dataloader: 
            sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item() # .detach()는 이미 no_grad() 컨텍스트이므로 불필요할 수 있으나, item()은 필요
            all_preds_batches.append(outputs.cpu().numpy())
            all_labels_batches.append(targets.cpu().numpy())
            
            # target_idxs_batch는 숫자형이므로 텐서로 반환될 가능성 높음
            if torch.is_tensor(target_idxs_batch):
                all_indices_batches.append(target_idxs_batch.cpu().numpy())
            else: # 안전장치
                all_indices_batches.append(np.array(target_idxs_batch))
            
            # target_trip_ids_batch는 문자열의 리스트임 (DataLoader가 그렇게 만듦)
            # .cpu().numpy() 대신 extend 사용
            all_trip_ids_collected.extend(target_trip_ids_batch) # 수정된 부분

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds_batches, axis=0)
    all_labels = np.concatenate(all_labels_batches, axis=0)
    all_indices = np.concatenate(all_indices_batches, axis=0)
    
    # 모든 trip ID를 모은 후 NumPy 배열로 변환
    all_trip_ids_np = np.array(all_trip_ids_collected) # 수정된 부분

    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    return avg_loss, rmse, all_preds, all_indices, all_trip_ids_np # 수정된 NumPy 배열 반환

# ----------------------------
# Optuna Objective 함수
# ----------------------------
def lstm_cv_objective(trial, train_trip_dfs_scaled, sequence_length):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    lstm_hidden_dim = trial.suggest_int('lstm_hidden_dim', 32, 256)
    lstm_num_layers = trial.suggest_int('lstm_num_layers', 1, 3)
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = 70 # Optuna는 수렴 속도도 중요하므로 적절히 설정
    patience = 10

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_val_rmse_list = []
    train_trip_dfs_array = np.array(train_trip_dfs_scaled, dtype=object)

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(train_trip_dfs_array)):
        print(f"--- Optuna Fold {fold_i+1}/5 ---")
        fold_train_dfs = train_trip_dfs_array[train_idx].tolist()
        fold_val_dfs = train_trip_dfs_array[val_idx].tolist()

        train_dataset = VehicleSequenceDataset(fold_train_dfs, sequence_length)
        val_dataset = VehicleSequenceDataset(fold_val_dfs, sequence_length)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"Skipping fold {fold_i+1} due to lack of sequences.")
            continue # 다음 폴드로

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) 
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model = LSTMModel(NUM_FEATURES, lstm_hidden_dim, lstm_num_layers, lstm_dropout).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        best_val_rmse = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochs):
            train_loss = train_epoch_lstm(model, train_loader, optimizer, criterion)
            val_loss, val_rmse, _, _, _ = evaluate_model_lstm(model, val_loader, criterion) 
            # print(f'Fold {fold_i+1}, Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}')

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
            
            trial.report(best_val_rmse, epoch)
            if trial.should_prune():
                print("Trial pruned by Optuna.")
                raise optuna.exceptions.TrialPruned()
        
        if best_val_rmse != float('inf'):
            fold_val_rmse_list.append(best_val_rmse)
        else:
            print(f"Fold {fold_i+1} did not produce a valid RMSE (remained inf).")
            
    if not fold_val_rmse_list:
        print("Warning: No valid folds completed for this trial. Returning high error.")
        return float('inf') # Pruning이 이 값을 사용하게 됨

    mean_cv_rmse = np.mean(fold_val_rmse_list)
    print(f"Trial {trial.number} finished. Mean CV RMSE: {mean_cv_rmse:.4f}")
    return mean_cv_rmse

def tune_lstm_hyperparameters(train_trip_dfs_scaled, selected_car, sequence_length, plot, n_trials=50):
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    objective_func = lambda trial: lstm_cv_objective(trial, train_trip_dfs_scaled, sequence_length)
    study.optimize(objective_func, n_trials=n_trials)

    print(f"Best trial for {selected_car} (LSTM): {study.best_trial.params}")
    print(f"Best CV RMSE (LSTM): {study.best_value}")

    if plot:
        # Optuna 결과 플로팅 (필요시 구현)
        pass
    
    if study.best_value == float('inf') or study.best_trial is None :
        print("Error: Optuna could not find valid parameters.")
        return None
    return study.best_trial.params

def train_final_lstm_model(train_trip_dfs_scaled, best_params, sequence_length):
    print("Training final LSTM model...")
    lstm_hidden_dim = best_params['lstm_hidden_dim']
    lstm_num_layers = best_params['lstm_num_layers']
    lstm_dropout = best_params['lstm_dropout']
    lr = best_params['lr']
    optimizer_name = best_params['optimizer']
    batch_size = best_params['batch_size']
    epochs = 100 

    train_dataset = VehicleSequenceDataset(train_trip_dfs_scaled, sequence_length)
    if len(train_dataset) == 0:
        print("Error: Cannot train final model, no sequences created.")
        return None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = LSTMModel(NUM_FEATURES, lstm_hidden_dim, lstm_num_layers, lstm_dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for sequences, targets, _, _ in train_loader: 
            sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item() # 수정: .detach() 추가
        avg_epoch_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Final Training Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}')
    end_time = time.time()
    print(f"Final LSTM training finished in {end_time - start_time:.2f} seconds.")
    return model

def integrate_and_compare(trip_data_df, pred_col_name='y_pred'):
    if trip_data_df.empty or len(trip_data_df) < 2:
        return 0, 0 

    trip_data_df = trip_data_df.sort_values(by='time').copy() 
    time_seconds = (trip_data_df['time'] - trip_data_df['time'].iloc[0]).dt.total_seconds().values

    if pred_col_name not in trip_data_df.columns:
        raise ValueError(f"Prediction column '{pred_col_name}' not found in trip_data_df.")
    if 'Power_phys' not in trip_data_df.columns:
        raise ValueError("'Power_phys' column not found in trip_data_df.")
    if 'Power_data' not in trip_data_df.columns:
        raise ValueError("'Power_data' column not found in trip_data_df.")

    trip_data_df['Power_hybrid'] = trip_data_df['Power_phys'] + trip_data_df[pred_col_name]

    hybrid_cum_integral = cumulative_trapezoid(trip_data_df['Power_hybrid'].values, time_seconds, initial=0)
    data_cum_integral = cumulative_trapezoid(trip_data_df['Power_data'].values, time_seconds, initial=0)
    
    hybrid_integral = hybrid_cum_integral[-1] if len(hybrid_cum_integral) > 0 else 0
    data_integral = data_cum_integral[-1] if len(data_cum_integral) > 0 else 0
    
    return hybrid_integral, data_integral

# ----------------------------
# LSTM 워크플로우 함수 (OOM 해결된 버전)
# ----------------------------
def run_lstm_workflow(vehicle_files_dict, selected_car_name, sequence_length=60, plot_flag=False, save_models_dir="models_lstm", existing_best_params=None):
    start_workflow_time = time.time()
    if selected_car_name not in vehicle_files_dict or not vehicle_files_dict[selected_car_name]:
        print(f"No files found for {selected_car_name}")
        return [], None

    files_list = vehicle_files_dict[selected_car_name]
    print(f"Starting LSTM workflow for {selected_car_name} with {len(files_list)} files (Seq Len: {sequence_length})...")

    train_files, test_files = train_test_split(files_list, test_size=0.2, random_state=42)
    print(f"Split: {len(train_files)} train files, {len(test_files)} test files.")

    print("Processing training files...")
    train_trip_dfs_raw = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file_lstm, f, trip_id=os.path.basename(f)) for i, f in enumerate(train_files)] # trip_id를 파일명으로 변경
        for future in as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                train_trip_dfs_raw.append(result)
    if not train_trip_dfs_raw:
        print(f"Error: No valid training data processed for {selected_car_name}.")
        return [], None

    print("Fitting scaler and scaling training data...")
    train_trip_dfs_s, fitted_scaler = scale_data_lstm(train_trip_dfs_raw)
    if fitted_scaler is None:
        print("Error: Scaler could not be fitted.")
        return [], None
    train_trip_dfs_s = [df for df in train_trip_dfs_s if df is not None and not df.empty]
    if not train_trip_dfs_s:
        print("Error: No valid scaled training data available.")
        return [], fitted_scaler

    print("Processing test files (raw data for MAPE)...")
    test_trip_dfs_raw = [] 
    with ProcessPoolExecutor() as executor: 
        futures = [executor.submit(process_single_file_lstm, f, trip_id=os.path.basename(f)) for f in test_files]
        for future in as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                test_trip_dfs_raw.append(result)
    
    # test_trip_dfs_raw가 비어있어도 다음 단계 진행은 가능 (MAPE 계산만 스킵)
    if not test_trip_dfs_raw:
        print(f"Warning: No valid raw test data processed for {selected_car_name}. MAPE calculation will be skipped.")

    print("Scaling test data (for model evaluation)...")
    # test_trip_dfs_raw가 비어있을 수 있으므로, 이 경우 test_trip_dfs_s도 비게 됨
    test_trip_dfs_s, _ = scale_data_lstm(test_trip_dfs_raw, fitted_scaler) 
    test_trip_dfs_s = [df for df in test_trip_dfs_s if df is not None and not df.empty]
    
    if existing_best_params is None:
        print("Starting hyperparameter tuning with Optuna...")
        tune_start_time = time.time()
        best_params = tune_lstm_hyperparameters(
            train_trip_dfs_s, selected_car_name, sequence_length, plot_flag, n_trials=30 
        )
        tune_end_time = time.time()
        if best_params is None:
            print("Hyperparameter tuning failed. Exiting.")
            return [], fitted_scaler
        print(f"Hyperparameter tuning finished in {tune_end_time - tune_start_time:.2f} seconds.")
    else:
        best_params = existing_best_params
        print(f"Using predefined best_params: {best_params}")

    final_model = train_final_lstm_model(train_trip_dfs_s, best_params, sequence_length)
    if final_model is None:
        print("Final model training failed.")
        return [], fitted_scaler

    print("Evaluating final model on the test set...")
    test_set_results = [{'rmse': float('nan'), 'test_mape': float('nan'), 'best_params': best_params, 'sequence_length': sequence_length}]

    if not test_trip_dfs_s: 
        print("Warning: No scaled test data available for RMSE evaluation.")
    else:
        test_dataset = VehicleSequenceDataset(test_trip_dfs_s, sequence_length)
        if len(test_dataset) == 0:
            print("Warning: No sequences created for test set. Cannot evaluate RMSE.")
        else:
            eval_batch_size = best_params.get('batch_size', 64) 
            test_loader = DataLoader(test_dataset, batch_size=eval_batch_size , shuffle=False, num_workers=0, pin_memory=True)
            _, test_rmse, test_preds_arr, test_indices_arr, test_trip_ids_arr = evaluate_model_lstm(final_model, test_loader, nn.MSELoss())
            print(f"Test RMSE (on Residual): {test_rmse:.4f}")
            test_set_results[0]['rmse'] = test_rmse

            if not test_trip_dfs_raw: 
                 print("Warning: No raw test data available for MAPE calculation (already noted).")
                 test_set_results[0]['test_mape'] = float('nan')
            else:
                predictions_map_df = pd.DataFrame({
                    'trip_id': test_trip_ids_arr.flatten(),
                    'original_index': test_indices_arr.flatten(),
                    'y_pred': test_preds_arr.flatten() 
                })

                hybrid_integrals_for_mape, data_integrals_for_mape = [], []
                processed_trips_count_mape = 0
                error_trips_count_mape = 0

                for original_trip_df in test_trip_dfs_raw: 
                    if original_trip_df is None or original_trip_df.empty or len(original_trip_df) < 2:
                        error_trips_count_mape += 1
                        continue
                    
                    current_trip_id = original_trip_df['trip_id'].iloc[0]
                    trip_specific_preds = predictions_map_df[predictions_map_df['trip_id'] == current_trip_id]
                    
                    mape_calc_df = original_trip_df.copy()
                    mape_calc_df = pd.merge(mape_calc_df, trip_specific_preds[['original_index', 'y_pred']], 
                                            on='original_index', how='left')
                    mape_calc_df['y_pred'] = mape_calc_df['y_pred'].fillna(0) 

                    try:
                        hybrid_e, data_e = integrate_and_compare(mape_calc_df, pred_col_name='y_pred')
                        if abs(data_e) > 1e-9: 
                            hybrid_integrals_for_mape.append(hybrid_e)
                            data_integrals_for_mape.append(data_e)
                            processed_trips_count_mape += 1
                        else: 
                            error_trips_count_mape +=1
                    except Exception as e:
                        print(f"Error during MAPE integration for trip {current_trip_id}: {e}")
                        error_trips_count_mape += 1
                
                if processed_trips_count_mape > 0:
                    mape_val_test = calculate_mape(np.array(data_integrals_for_mape), np.array(hybrid_integrals_for_mape))
                else:
                    mape_val_test = float('nan')
                
                print(f"\nTest Set Integration Metrics ({processed_trips_count_mape} trips for MAPE, {error_trips_count_mape} trips skipped/error):")
                print(f"MAPE (Energy): {mape_val_test:.2f}%" if not np.isnan(mape_val_test) else "MAPE: Not Available")
                test_set_results[0]['test_mape'] = mape_val_test

    if save_models_dir:
        if not os.path.exists(save_models_dir):
            os.makedirs(save_models_dir)
        if final_model:
            model_file_path = os.path.join(save_models_dir, f"LSTM_best_model_{selected_car_name}_seq{sequence_length}.pth")
            try:
                torch.save(final_model.state_dict(), model_file_path)
                print(f"Best LSTM model state_dict saved to {model_file_path}")
            except Exception as e:
                print(f"Error saving LSTM model state_dict: {e}")
        if fitted_scaler:
            scaler_file_path = os.path.join(save_models_dir, f'LSTM_scaler_{selected_car_name}_seq{sequence_length}.pkl')
            try:
                with open(scaler_file_path, 'wb') as f_scaler:
                    pickle.dump(fitted_scaler, f_scaler)
                print(f"Scaler saved at {scaler_file_path}")
            except Exception as e:
                print(f"Error saving scaler: {e}")

    end_workflow_time = time.time()
    print(f"LSTM workflow for {selected_car_name} completed in {end_workflow_time - start_workflow_time:.2f} seconds.")
    return test_set_results, fitted_scaler