# GS_Train_LSTM.py
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
# Composite contour plot might not be suitable/easy for LSTM interpretation
# from GS_plot import plot_composite_contour
from concurrent.futures import ProcessPoolExecutor, as_completed
from optuna.trial import TrialState
import time

# ----------------------------
# 전역 변수 / 상수 정의
# ----------------------------
# 기본 특성에 대한 상수만 사용
SPEED_MIN = 0 / 3.6
SPEED_MAX = 230 / 3.6
ACCELERATION_MIN = -15
ACCELERATION_MAX = 9
TEMP_MIN = -30
TEMP_MAX = 50

# LSTM 입력 특성 정의
FEATURE_COLS = ['speed', 'acceleration', 'ext_temp']
NUM_FEATURES = len(FEATURE_COLS)

# CUDA 사용 가능 여부 확인 및 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----------------------------
# 데이터 처리 함수
# ----------------------------
def process_single_file_lstm(file, trip_id):
    """
    파일 하나를 읽어 LSTM에 필요한 기본 특성과 타겟(Residual)만 포함하여 반환.
    롤링 특성 계산 제거.
    """
    try:
        data = pd.read_csv(file)
        # Power_phys와 Power_data가 있는지 확인
        if 'Power_phys' in data.columns and 'Power_data' in data.columns and all(c in data.columns for c in FEATURE_COLS):
            data['Residual'] = data['Power_data'] - data['Power_phys']
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
            data['trip_id'] = trip_id

            # LSTM에 필요한 컬럼만 선택 + 원래 index 보존
            required_cols = FEATURE_COLS + ['Residual', 'trip_id', 'time', 'Power_phys', 'Power_data']
            # 원래 DataFrame index를 컬럼으로 추가하여 나중에 예측값 매핑에 사용
            data['original_index'] = data.index
            required_cols.append('original_index')

            return data[required_cols]
        else:
            missing = [c for c in FEATURE_COLS + ['Power_phys', 'Power_data'] if c not in data.columns]
            # print(f"Warning: Missing required columns {missing} in {file}. Skipping.")
            return None
    except Exception as e:
        print(f"Error processing file {file} for LSTM: {e}")
    return None

def scale_data_lstm(df_list, scaler=None):
    """
    여러 trip DataFrame 리스트를 받아 스케일링.
    FEATURE_COLS에 대해서만 MinMaxScaling.
    """
    # 스케일러 학습 (scaler가 None일 경우)
    if scaler is None:
        # 모든 DataFrame에서 스케일링할 컬럼 데이터만 추출하여 합치기
        combined_features = pd.concat([df[FEATURE_COLS] for df in df_list if df is not None and not df.empty], ignore_index=True)

        if combined_features.empty:
            print("Error: No data available to fit the scaler.")
            return df_list, None # 스케일러 학습 불가

        # 기본 특성에 대한 Min/Max 값 정의 (더미 데이터 대신 실제 데이터 사용 권장하나, 여기선 상수 사용 유지)
        min_vals = [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN]
        max_vals = [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX]
        dummy_df_min = pd.DataFrame([min_vals], columns=FEATURE_COLS)
        dummy_df_max = pd.DataFrame([max_vals], columns=FEATURE_COLS)
        dummy_df = pd.concat([dummy_df_min, dummy_df_max], ignore_index=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        # scaler.fit(combined_features) # 실제 데이터로 학습하는 것이 더 정확함
        scaler.fit(dummy_df) # 여기서는 이전 방식 유지 (상수 기반)
        print("Scaler fitted.")

    # 스케일링 적용
    scaled_df_list = []
    for df in df_list:
        if df is not None and not df.empty:
            df_copy = df.copy()
            # FEATURE_COLS 순서에 맞게 데이터 추출 후 스케일링
            features_to_scale = df_copy[FEATURE_COLS]
            scaled_features = scaler.transform(features_to_scale)
            df_copy[FEATURE_COLS] = scaled_features
            scaled_df_list.append(df_copy)
        else:
            scaled_df_list.append(None) # 빈 DataFrame이나 None은 그대로 유지

    return scaled_df_list, scaler

def create_sequences(data, sequence_length):
    """ 단일 trip DataFrame에서 LSTM용 시퀀스 생성 """
    xs, ys, idxs = [], [], []
    # .values로 numpy 배열로 변환하여 속도 향상
    feature_values = data[FEATURE_COLS].values
    target_values = data['Residual'].values
    original_indices = data['original_index'].values # 원래 인덱스 가져오기

    if len(data) >= sequence_length:
        for i in range(len(data) - sequence_length + 1):
            start_idx = i
            end_idx = i + sequence_length
            # 시퀀스 입력: [seq_len, num_features]
            sequence_x = feature_values[start_idx:end_idx]
            # 타겟: 시퀀스 마지막 시점의 Residual 값 [1]
            sequence_y = target_values[end_idx - 1]
            # 타겟에 해당하는 원래 DataFrame 인덱스
            target_idx = original_indices[end_idx - 1]

            xs.append(sequence_x)
            ys.append(sequence_y)
            idxs.append(target_idx) # 인덱스 저장

    # Numpy 배열로 변환하여 반환
    return np.array(xs), np.array(ys), np.array(idxs)

# ----------------------------
# PyTorch Dataset (Sequence) 및 LSTM 모델 정의
# ----------------------------
class VehicleSequenceDataset(Dataset):
    def __init__(self, scaled_trip_df_list, sequence_length):
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        self.target_indices = [] # 타겟 값의 원래 인덱스 저장

        print(f"Creating sequences with length {sequence_length}...")
        num_processed = 0
        total_sequences = 0
        for df in scaled_trip_df_list:
            if df is not None and not df.empty and len(df) >= sequence_length:
                seqs, targs, idxs = create_sequences(df, sequence_length)
                if len(seqs) > 0:
                    self.sequences.append(seqs)
                    self.targets.append(targs)
                    self.target_indices.append(idxs) # 인덱스 추가
                    num_processed += 1
                    total_sequences += len(seqs)
            # else: # Debugging short trips
                 # if df is not None: print(f"Trip too short ({len(df)} points), skipping sequence creation.")

        if not self.sequences:
             print("Warning: No sequences could be created from the provided data.")
             # 데이터를 찾을 수 없을 때 빈 리스트를 사용하도록 초기화
             self.sequences = np.empty((0, sequence_length, NUM_FEATURES))
             self.targets = np.empty((0,))
             self.target_indices = np.empty((0,), dtype=int)
        else:
             # 리스트들을 numpy 배열로 결합
             self.sequences = np.concatenate(self.sequences, axis=0)
             self.targets = np.concatenate(self.targets, axis=0)
             self.target_indices = np.concatenate(self.target_indices, axis=0) # 인덱스 결합

        print(f"Processed {num_processed} trips, created {total_sequences} sequences.")

    def __len__(self):
        # Return 0 if sequences is not a numpy array or empty
        return len(self.targets) if isinstance(self.targets, np.ndarray) else 0


    def __getitem__(self, idx):
        # idx가 범위를 벗어나는 경우 처리 (필수는 아닐 수 있음)
        if idx >= len(self.targets):
            raise IndexError("Index out of bounds")

        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32).view(1) # [1] 형태로 변환
        target_idx = self.target_indices[idx] # 해당 타겟의 원본 인덱스 반환

        return sequence, target, target_idx # 원본 인덱스도 반환

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # batch_first=True: 입력/출력 텐서를 (batch, seq, feature) 형태로 받음
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        # LSTM 마지막 time step의 출력을 받아 Residual(1개 값) 예측
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 초기 hidden state와 cell state는 0으로 설정 (LSTM이 자동으로 처리)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM 순전파
        # out: 각 time step의 LSTM 출력 (batch, seq_len, hidden_dim)
        # (hn, cn): 마지막 time step의 hidden state와 cell state
        out, _ = self.lstm(x) #, (h0, c0))

        # 마지막 time step의 출력만 사용 (batch, hidden_dim)
        out = out[:, -1, :]

        # Fully connected layer 통과
        out = self.fc(out)
        return out

# ----------------------------
# 학습 및 평가 루프 (LSTM 용)
# ----------------------------
def train_epoch_lstm(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    # DataLoader는 (sequence, target, target_idx) 반환
    for sequences, targets, _ in dataloader: # target_idx는 학습에 사용 안 함
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
    all_preds = []
    all_labels = []
    all_indices = [] # 예측값에 해당하는 원본 인덱스 저장
    with torch.no_grad():
         # DataLoader는 (sequence, target, target_idx) 반환
        for sequences, targets, target_idxs in dataloader:
            sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
            all_indices.append(target_idxs.cpu().numpy()) # 인덱스 저장

    avg_loss = total_loss / len(dataloader)
    # 예측값, 실제값, 인덱스 결합
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_indices = np.concatenate(all_indices, axis=0) # 인덱스 결합

    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    # 평가 결과로 rmse와 함께 예측값/인덱스 반환 (MAPE 계산 위해)
    return avg_loss, rmse, all_preds, all_indices

# ----------------------------
# Optuna Objective 함수 (LSTM 용)
# ----------------------------
def lstm_cv_objective(trial, train_trip_dfs_scaled, scaler, sequence_length):
    # 하이퍼파라미터 샘플링 (LSTM 용)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True) # LSTM은 좀 더 작은 lr 필요할 수 있음
    lstm_hidden_dim = trial.suggest_int('lstm_hidden_dim', 32, 256)
    lstm_num_layers = trial.suggest_int('lstm_num_layers', 1, 3)
    # LSTM 내부 드롭아웃 (layer > 1 일 때 적용됨)
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128]) # 시퀀스 데이터는 메모리 더 사용

    epochs = 70 # LSTM은 수렴에 더 많은 에폭 필요할 수 있으나, 조기 종료 사용
    patience = 10

    # K-Fold는 파일 목록 대신 DataFrame 리스트의 인덱스에 적용
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_val_rmse_list = []
    # DataFrame 리스트를 numpy 배열로 변환하여 인덱싱 용이하게
    train_trip_dfs_array = np.array(train_trip_dfs_scaled, dtype=object)

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(train_trip_dfs_array)):
        print(f"--- Optuna Fold {fold_i+1}/5 ---")
        fold_train_dfs = train_trip_dfs_array[train_idx].tolist()
        fold_val_dfs = train_trip_dfs_array[val_idx].tolist()

        # Prepare Datasets and DataLoaders using sequence dataset
        train_dataset = VehicleSequenceDataset(fold_train_dfs, sequence_length)
        val_dataset = VehicleSequenceDataset(fold_val_dfs, sequence_length)

        # 데이터셋 생성 실패 시 fold 건너뛰기
        if len(train_dataset) == 0 or len(val_dataset) == 0:
             print(f"Skipping fold {fold_i+1} due to lack of sequences.")
             # 실패한 fold에 대해 높은 RMSE 값 반환 (또는 다른 처리)
             # fold_val_rmse_list.append(float('inf')) # 무한대 추가는 평균 계산에 문제
             continue # 이 fold 건너뛰고 다음 fold 진행

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # Model, Criterion, Optimizer
        model = LSTMModel(NUM_FEATURES, lstm_hidden_dim, lstm_num_layers, lstm_dropout).to(DEVICE)
        criterion = nn.MSELoss()
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr)

        # Training loop with early stopping
        best_val_rmse = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochs):
            train_loss = train_epoch_lstm(model, train_loader, optimizer, criterion)
            # 평가 함수는 loss, rmse, preds, indices 반환
            val_loss, val_rmse, _, _ = evaluate_model_lstm(model, val_loader, criterion)
            print(f'Fold {fold_i+1}, Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}')

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                epochs_no_improve = 0
                # Optionally save the best model state for this fold here
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

            # Optuna Pruning
            trial.report(best_val_rmse, epoch)
            if trial.should_prune():
                print("Trial pruned by Optuna.")
                # 에러 대신 Pruned 예외 발생
                raise optuna.exceptions.TrialPruned()

        # 유효한 RMSE가 계산된 경우에만 리스트에 추가
        if best_val_rmse != float('inf'):
             fold_val_rmse_list.append(best_val_rmse)
        else:
             print(f"Fold {fold_i+1} did not produce a valid RMSE.")


    # 유효한 fold 결과가 하나 이상 있을 때만 평균 계산
    if not fold_val_rmse_list:
         print("Warning: No valid folds completed. Returning high error.")
         return float('inf') # 모든 fold 실패 시 높은 값 반환

    mean_cv_rmse = np.mean(fold_val_rmse_list)
    print(f"Trial {trial.number} finished. Mean CV RMSE: {mean_cv_rmse:.4f}")
    return mean_cv_rmse


def tune_lstm_hyperparameters(train_trip_dfs_scaled, scaler, selected_car, sequence_length, plot, n_trials=50):
    """ Optuna 스터디 실행 (LSTM용) """
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    objective_func = lambda trial: lstm_cv_objective(trial, train_trip_dfs_scaled, scaler, sequence_length)
    study.optimize(objective_func, n_trials=n_trials)

    print(f"Best trial for {selected_car} (LSTM): {study.best_trial.params}")
    print(f"Best CV RMSE (LSTM): {study.best_value}")

    if plot:
        # --- Plotting logic (DNN과 유사하게 Optuna 결과 플롯) ---
        # ... (DNN 버전의 Optuna 결과 플로팅 코드와 거의 동일하게 구현) ...
        trials_df = study.trials_dataframe()
        # ... 저장 및 플롯 표시 ...
        print("Optuna results plot generated (if complete trials exist).")


    # 최적 파라미터 반환 (best_value가 유효하지 않으면 None 반환 등 예외처리 추가 가능)
    if study.best_value == float('inf'):
        print("Error: Optuna could not find valid parameters.")
        return None
    return study.best_trial.params


def train_final_lstm_model(train_trip_dfs_scaled, best_params, sequence_length):
    """ 최종 LSTM 모델 학습 """
    print("Training final LSTM model...")
    # best_params에서 LSTM 파라미터 추출
    lstm_hidden_dim = best_params['lstm_hidden_dim']
    lstm_num_layers = best_params['lstm_num_layers']
    lstm_dropout = best_params['lstm_dropout']
    lr = best_params['lr']
    optimizer_name = best_params['optimizer']
    batch_size = best_params['batch_size']
    epochs = 100 # 최종 학습 에폭 (튜닝 결과 참고하여 조정 가능)

    # 전체 학습 데이터셋 생성
    train_dataset = VehicleSequenceDataset(train_trip_dfs_scaled, sequence_length)
    if len(train_dataset) == 0:
        print("Error: Cannot train final model, no sequences created for training.")
        return None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # 모델 초기화
    model = LSTMModel(NUM_FEATURES, lstm_hidden_dim, lstm_num_layers, lstm_dropout).to(DEVICE)
    criterion = nn.MSELoss()
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 학습 루프
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for sequences, targets, _ in train_loader: # 인덱스는 사용 안 함
            sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() # detach 필요 없음 (loss.backward() 후)
        avg_epoch_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
             print(f'Final Training Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}')
    end_time = time.time()
    print(f"Final LSTM training finished in {end_time - start_time:.2f} seconds.")
    return model

# ----------------------------
# LSTM 워크플로우 함수
# ----------------------------
def run_lstm_workflow(vehicle_files, selected_car, sequence_length=60, plot=False, save_dir="models_lstm", predefined_best_params=None):
    """ LSTM 모델 학습 및 평가 워크플로우 """
    start_workflow_time = time.time()
    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
        print(f"No files found for {selected_car}")
        return [], None

    files = vehicle_files[selected_car]
    print(f"Starting LSTM workflow for {selected_car} with {len(files)} files (Sequence Length: {sequence_length})...")

    # 1. 파일 단위 Train/Test 분할
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    print(f"Split: {len(train_files)} train files, {len(test_files)} test files.")

    # 2. Train 데이터 처리 (DataFrame 리스트로 유지)
    print("Processing training files...")
    train_trip_dfs = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file_lstm, f, trip_id=i) for i, f in enumerate(train_files)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                train_trip_dfs.append(result)

    if not train_trip_dfs:
        print(f"Error: No valid training data could be processed for {selected_car}.")
        return [], None

    # 3. Scaler Fit 및 Train 데이터 스케일링 (DataFrame 리스트 입력/출력)
    print("Fitting scaler and scaling training data...")
    train_trip_dfs_scaled, scaler = scale_data_lstm(train_trip_dfs)
    if scaler is None:
         print("Error: Scaler could not be fitted.")
         return [], None
    # 스케일링 후 비어있는 df 제거
    train_trip_dfs_scaled = [df for df in train_trip_dfs_scaled if df is not None and not df.empty]
    if not train_trip_dfs_scaled:
        print("Error: No valid scaled training data available.")
        return [], scaler


    # 4. Test 데이터 처리 (DataFrame 리스트로 유지)
    print("Processing test files...")
    test_trip_dfs = []
    original_test_data_map = {} # MAPE 계산 위해 원본 데이터 저장 (인덱스 기준)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file_lstm, f, trip_id=1000 + j) for j, f in enumerate(test_files)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                test_trip_dfs.append(result)
                # 원래 데이터 저장 (original_index를 key로 사용)
                for idx, row in result.iterrows():
                    original_test_data_map[row['original_index']] = row

    if not test_trip_dfs:
        print(f"Warning: No valid test data processed for {selected_car}. Cannot evaluate.")
        return [], scaler

    # 5. Test 데이터 스케일링 (DataFrame 리스트 입력/출력)
    print("Scaling test data...")
    test_trip_dfs_scaled, _ = scale_data_lstm(test_trip_dfs, scaler)
    # 스케일링 후 비어있는 df 제거
    test_trip_dfs_scaled = [df for df in test_trip_dfs_scaled if df is not None and not df.empty]
    if not test_trip_dfs_scaled:
        print("Warning: No valid scaled test data available for evaluation.")
        # 평가 없이 종료하거나 다른 처리
        return [], scaler

    # -----------------------
    # 6. Hyperparameter Tuning or Use Predefined
    # -----------------------
    if predefined_best_params is None:
        print("Starting hyperparameter tuning with Optuna...")
        tune_start_time = time.time()
        best_params = tune_lstm_hyperparameters(
            train_trip_dfs_scaled, scaler, selected_car, sequence_length, plot, n_trials=30 # LSTM은 오래 걸리므로 trial 수 줄임
        )
        tune_end_time = time.time()
        if best_params is None: # 튜닝 실패 시
             print("Hyperparameter tuning failed. Exiting.")
             return [], scaler
        print(f"Hyperparameter tuning finished in {tune_end_time - tune_start_time:.2f} seconds.")
    else:
        best_params = predefined_best_params
        print(f"Using predefined best_params: {best_params}")

    # -----------------------
    # 7. 최종 모델 학습
    # -----------------------
    final_model = train_final_lstm_model(train_trip_dfs_scaled, best_params, sequence_length)
    if final_model is None:
         print("Final model training failed.")
         return [], scaler

    # -----------------------
    # 8. Test Set 평가
    # -----------------------
    print("Evaluating final model on the test set...")
    # 테스트 데이터셋 생성
    test_dataset = VehicleSequenceDataset(test_trip_dfs_scaled, sequence_length)
    if len(test_dataset) == 0:
        print("Warning: No sequences created for test set. Cannot evaluate.")
        # 결과 없이 종료
        results = [{'rmse': float('nan'), 'test_mape': float('nan'), 'best_params': best_params}]
        # 모델 저장은 가능
    else:
        test_loader = DataLoader(test_dataset, batch_size=best_params.get('batch_size', 64) * 2, shuffle=False, num_workers=2, pin_memory=True)

        # 평가 함수 호출 (loss, rmse, preds, indices 반환)
        _, test_rmse, test_preds, test_indices = evaluate_model_lstm(final_model, test_loader, nn.MSELoss())
        print(f"Test RMSE (on Residual): {test_rmse:.4f}")

        # --- MAPE 계산 ---
        # 예측값을 원래 시간 인덱스에 매핑
        # test_preds는 (num_sequences, 1) 형태, flatten() 필요
        predictions_df = pd.DataFrame({'lstm_pred': test_preds.flatten()}, index=test_indices)

        # 원본 test 데이터 DataFrame들에 예측값 합치기
        merged_test_dfs = []
        for df in test_trip_dfs: # 스케일링 안된 원본 사용
             if df is not None and not df.empty:
                 # original_index를 기준으로 예측값 join
                 df_merged = df.join(predictions_df, on='original_index')
                 # 예측값이 없는 행(시퀀스 시작 부분)은 0으로 채우거나 다른 전략 사용 가능
                 # 여기서는 일단 NaN으로 두고 integrate_and_compare에서 처리 기대 (또는 여기서 처리)
                 df_merged['lstm_pred'].fillna(0, inplace=True) # 예측 없는 부분 0으로 가정
                 merged_test_dfs.append(df_merged)


        hybrid_integrals_test, data_integrals_test = [], []
        processed_trips_mape = 0
        integration_errors_mape = 0

        # Trip별로 그룹화하지 않고, 준비된 merged_test_dfs 리스트 사용
        trip_groups_for_mape = pd.concat(merged_test_dfs).groupby('trip_id')

        for trip_id, group in trip_groups_for_mape:
             if len(group) < 2: continue
             try:
                  # integrate_and_compare 함수는 'dnn_pred' 대신 'lstm_pred' 사용하도록 수정 필요
                  # 여기서는 임시로 컬럼명 변경 또는 함수 수정 가정
                  group_renamed = group.rename(columns={'lstm_pred': 'dnn_pred'}) # 임시 이름 변경
                  hybrid_integral, data_integral = integrate_and_compare(group_renamed)

                  if abs(data_integral) > 1e-6:
                       hybrid_integrals_test.append(hybrid_integral)
                       data_integrals_test.append(data_integral)
                       processed_trips_mape += 1
             except Exception as e:
                  print(f"Error during MAPE integration for trip {trip_id}: {e}")
                  integration_errors_mape += 1

        mape_test = calculate_mape(np.array(data_integrals_test), np.array(hybrid_integrals_test)) if processed_trips_mape > 0 else float('nan')

        print(f"\nTest Set Integration Metrics ({processed_trips_mape} trips processed, {integration_errors_mape} errors):")
        print(f"MAPE (Energy): {mape_test:.2f}%" if not np.isnan(mape_test) else "MAPE: Not Available")

        results = [{
            'rmse': test_rmse,
            'test_mape': mape_test,
            'best_params': best_params,
            'sequence_length': sequence_length
        }]

    # -----------------------
    # 9. 모델 및 스케일러 저장
    # -----------------------
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save Model State Dictionary
        model_file = os.path.join(save_dir, f"LSTM_best_model_{selected_car}_seq{sequence_length}.pth")
        try:
            if final_model: # 모델 학습 성공 시 저장
                torch.save(final_model.state_dict(), model_file)
                print(f"Best LSTM model state_dict saved to {model_file}")
        except Exception as e:
            print(f"Error saving LSTM model state_dict: {e}")

        # Save Scaler
        scaler_path = os.path.join(save_dir, f'LSTM_scaler_{selected_car}_seq{sequence_length}.pkl')
        try:
            if scaler: # 스케일러 생성 성공 시 저장
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"Scaler saved at {scaler_path}")
        except Exception as e:
            print(f"Error saving scaler: {e}")

    end_workflow_time = time.time()
    print(f"LSTM workflow for {selected_car} completed in {end_workflow_time - start_workflow_time:.2f} seconds.")

    return results, scaler