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

# GS_Functions.py 에서 calculate_mape를 가져온다고 가정
# GS_plot.py 에서 plot_composite_contour 등을 가져온다고 가정 (여기서는 직접 사용하지 않음)
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
    xs, ys, idxs, trip_ids_for_seq = [], [], [], [] # trip_id도 함께 저장
    feature_values = data[FEATURE_COLS].values
    target_values = data['Residual'].values
    original_indices = data['original_index'].values
    trip_id_value = data['trip_id'].iloc[0] # 해당 DataFrame의 trip_id

    if len(data) >= sequence_length:
        for i in range(len(data) - sequence_length + 1):
            sequence_x = feature_values[i : i + sequence_length]
            sequence_y = target_values[i + sequence_length - 1]
            target_idx = original_indices[i + sequence_length - 1]
            
            xs.append(sequence_x)
            ys.append(sequence_y)
            idxs.append(target_idx)
            trip_ids_for_seq.append(trip_id_value) # 각 시퀀스의 타겟에 해당하는 trip_id 저장

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
        self.target_trip_ids_data = [] # 각 타겟에 해당하는 trip_id 저장

        print(f"Creating sequences with length {sequence_length}...")
        num_processed = 0
        total_sequences = 0
        for df in scaled_trip_df_list:
            if df is not None and not df.empty and len(df) >= sequence_length:
                seqs, targs, idxs, t_ids = create_sequences(df, sequence_length)
                if len(seqs) > 0:
                    self.sequences_data.append(seqs)
                    self.targets_data.append(targs)
                    self.target_indices_data.append(idxs)
                    self.target_trip_ids_data.append(t_ids) # trip_id 저장
                    num_processed += 1
                    total_sequences += len(seqs)

        if not self.sequences_data:
            print("Warning: No sequences could be created.")
            self.sequences = np.empty((0, sequence_length, NUM_FEATURES))
            self.targets = np.empty((0,))
            self.target_indices = np.empty((0,), dtype=int)
            self.target_trip_ids = np.empty((0,), dtype=type(trip_id_value) if 'trip_id_value' in locals() else int) # trip_id 타입 주의
        else:
            self.sequences = np.concatenate(self.sequences_data, axis=0)
            self.targets = np.concatenate(self.targets_data, axis=0)
            self.target_indices = np.concatenate(self.target_indices_data, axis=0)
            self.target_trip_ids = np.concatenate(self.target_trip_ids_data, axis=0) # trip_id 결합
        print(f"Processed {num_processed} trips, created {total_sequences} sequences.")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if idx >= len(self.targets):
            raise IndexError("Index out of bounds")
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32).view(1)
        target_idx = self.target_indices[idx]
        target_trip_id = self.target_trip_ids[idx] # trip_id 반환
        return sequence, target, target_idx, target_trip_id # trip_id도 반환

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
    for sequences, targets, _, _ in dataloader: # target_idx, target_trip_id는 학습에 미사용
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
    all_preds, all_labels, all_indices, all_trip_ids = [], [], [], []
    with torch.no_grad():
        for sequences, targets, target_idxs, target_trip_ids_batch in dataloader: # trip_id도 받음
            sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
            all_indices.append(target_idxs.cpu().numpy())
            all_trip_ids.append(target_trip_ids_batch.cpu().numpy()) # trip_id 저장

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    all_trip_ids = np.concatenate(all_trip_ids, axis=0) # trip_id 결합

    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    return avg_loss, rmse, all_preds, all_indices, all_trip_ids # trip_id 반환 추가

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
    epochs = 70
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
            continue

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) # num_workers=0 for potential memory saving
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model = LSTMModel(NUM_FEATURES, lstm_hidden_dim, lstm_num_layers, lstm_dropout).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        best_val_rmse = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochs):
            train_loss = train_epoch_lstm(model, train_loader, optimizer, criterion)
            val_loss, val_rmse, _, _, _ = evaluate_model_lstm(model, val_loader, criterion) # 추가 반환값 _ 처리
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
            print(f"Fold {fold_i+1} did not produce a valid RMSE.")
            
    if not fold_val_rmse_list: # 모든 폴드가 실패했거나 유효한 RMSE가 없는 경우
        print("Warning: No valid folds completed for this trial. Returning high error.")
        return float('inf')

    mean_cv_rmse = np.mean(fold_val_rmse_list)
    print(f"Trial {trial.number} finished. Mean CV RMSE: {mean_cv_rmse:.4f}")
    return mean_cv_rmse

def tune_lstm_hyperparameters(train_trip_dfs_scaled, selected_car, sequence_length, plot, n_trials=50):
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    objective_func = lambda trial: lstm_cv_objective(trial, train_trip_dfs_scaled, sequence_length) # scaler는 objective 내부에서 사용 안함
    study.optimize(objective_func, n_trials=n_trials)

    print(f"Best trial for {selected_car} (LSTM): {study.best_trial.params}")
    print(f"Best CV RMSE (LSTM): {study.best_value}")

    if plot:
        # Optuna 결과 플로팅 (생략, 필요시 DNN 버전 참고)
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
    epochs = 100 # 최종 학습 에폭

    train_dataset = VehicleSequenceDataset(train_trip_dfs_scaled, sequence_length)
    if len(train_dataset) == 0:
        print("Error: Cannot train final model, no sequences created.")
        return None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) # num_workers=0

    model = LSTMModel(NUM_FEATURES, lstm_hidden_dim, lstm_num_layers, lstm_dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for sequences, targets, _, _ in train_loader: # target_idx, target_trip_id 미사용
            sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Final Training Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}')
    end_time = time.time()
    print(f"Final LSTM training finished in {end_time - start_time:.2f} seconds.")
    return model

# MAPE 계산을 위한 에너지 적분 함수 (XGBoost 코드의 함수와 호환되도록 정의)
# GS_Functions.py에 있거나, 여기서 직접 정의
def integrate_and_compare(trip_data_df, pred_col_name='y_pred'):
    """
    단일 트립 DataFrame을 받아 Power_data와 Power_hybrid (Power_phys + pred_col_name)의
    누적 에너지를 계산하고 반환.
    'time' 컬럼은 datetime 객체여야 함.
    'Power_phys', pred_col_name, 'Power_data' 컬럼이 필요.
    """
    if trip_data_df.empty or len(trip_data_df) < 2:
        return 0, 0 # 또는 (np.nan, np.nan) 등 오류 값

    # 시간 순 정렬 및 초 단위 변환
    trip_data_df = trip_data_df.sort_values(by='time').copy() # 복사본 사용
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
# LSTM 워크플로우 함수 (OOM 해결)
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
        futures = [executor.submit(process_single_file_lstm, f, trip_id=i) for i, f in enumerate(train_files)]
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
    test_trip_dfs_raw = [] # 원본 테스트 데이터 (스케일링X, MAPE 계산용)
    with ProcessPoolExecutor() as executor: # trip_id는 파일명 또는 고유 ID로 설정하는 것이 더 좋음
        futures = [executor.submit(process_single_file_lstm, f, trip_id=os.path.basename(f)) for f in test_files]
        for future in as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                test_trip_dfs_raw.append(result)
    if not test_trip_dfs_raw:
        print(f"Warning: No valid test data processed for {selected_car_name}. Cannot evaluate fully.")
        # 스케일러만 반환하거나 빈 결과 반환
        # return [], fitted_scaler 

    print("Scaling test data (for model evaluation)...")
    test_trip_dfs_s, _ = scale_data_lstm(test_trip_dfs_raw, fitted_scaler) # 원본 리스트를 스케일링
    test_trip_dfs_s = [df for df in test_trip_dfs_s if df is not None and not df.empty]
    
    # Hyperparameter Tuning or Use Predefined
    if existing_best_params is None:
        print("Starting hyperparameter tuning with Optuna...")
        tune_start_time = time.time()
        best_params = tune_lstm_hyperparameters(
            train_trip_dfs_s, selected_car_name, sequence_length, plot_flag, n_trials=30 # trial 수 조정
        )
        tune_end_time = time.time()
        if best_params is None:
            print("Hyperparameter tuning failed. Exiting.")
            return [], fitted_scaler
        print(f"Hyperparameter tuning finished in {tune_end_time - tune_start_time:.2f} seconds.")
    else:
        best_params = existing_best_params
        print(f"Using predefined best_params: {best_params}")

    # 최종 모델 학습
    final_model = train_final_lstm_model(train_trip_dfs_s, best_params, sequence_length)
    if final_model is None:
        print("Final model training failed.")
        return [], fitted_scaler

    # Test Set 평가
    print("Evaluating final model on the test set...")
    test_set_results = [{'rmse': float('nan'), 'test_mape': float('nan'), 'best_params': best_params, 'sequence_length': sequence_length}]

    if not test_trip_dfs_s: # 스케일링된 테스트 데이터가 없는 경우
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

            # --- MAPE 계산 (OOM 해결된 방식) ---
            # 예측값을 (trip_id, original_index) 기준으로 매핑할 수 있는 DataFrame 생성
            # test_preds_arr: 예측된 잔차값 배열
            # test_indices_arr: 각 잔차값에 해당하는 원본 DataFrame 내 original_index 배열
            # test_trip_ids_arr: 각 잔차값에 해당하는 trip_id 배열
            predictions_map_df = pd.DataFrame({
                'trip_id': test_trip_ids_arr.flatten(),
                'original_index': test_indices_arr.flatten(),
                'y_pred': test_preds_arr.flatten() # 컬럼명을 'y_pred'로 하여 integrate_and_compare와 호환
            })

            hybrid_integrals_for_mape, data_integrals_for_mape = [], []
            processed_trips_count_mape = 0
            error_trips_count_mape = 0

            if not test_trip_dfs_raw: # 원본 테스트 데이터가 없는 경우 MAPE 계산 불가
                 print("Warning: No raw test data available for MAPE calculation.")
                 test_set_results[0]['test_mape'] = float('nan')
            else:
                for original_trip_df in test_trip_dfs_raw: # 스케일링 안된 원본 DataFrame 리스트 사용
                    if original_trip_df is None or original_trip_df.empty or len(original_trip_df) < 2:
                        error_trips_count_mape += 1
                        continue
                    
                    current_trip_id = original_trip_df['trip_id'].iloc[0]
                    
                    # 해당 trip_id에 대한 예측값만 필터링
                    trip_specific_preds = predictions_map_df[predictions_map_df['trip_id'] == current_trip_id]
                    
                    # 원본 트립 데이터에 예측값(y_pred) 병합
                    # original_index를 기준으로 join
                    # copy()를 사용하여 SettingWithCopyWarning 방지
                    mape_calc_df = original_trip_df.copy()
                    mape_calc_df = pd.merge(mape_calc_df, trip_specific_preds[['original_index', 'y_pred']], 
                                            on='original_index', how='left')
                    
                    # LSTM 예측은 시퀀스 길이만큼 앞부분에 NaN이 있을 수 있음. 0으로 채움.
                    # 또는 다른 전략(예: 해당 지점 예측 안 함) 사용 가능
                    mape_calc_df['y_pred'] = mape_calc_df['y_pred'].fillna(0) 

                    try:
                        # integrate_and_compare 함수는 'y_pred' 컬럼을 사용
                        hybrid_e, data_e = integrate_and_compare(mape_calc_df, pred_col_name='y_pred')
                        if abs(data_e) > 1e-9: # 실제 에너지가 0에 매우 가까운 경우 제외 (0으로 나누기 방지)
                            hybrid_integrals_for_mape.append(hybrid_e)
                            data_integrals_for_mape.append(data_e)
                            processed_trips_count_mape += 1
                        else: # 실제 에너지가 0인 트립은 MAPE 계산에서 제외
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

    # 모델 및 스케일러 저장
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