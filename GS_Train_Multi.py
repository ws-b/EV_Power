import os
import numpy as np
import random
import matplotlib.pyplot as plt
from threading import Thread
from GS_Functions import compute_rmse
from GS_Train_XGboost import cross_validate as xgb_cross_validate
from GS_Train_Only_XGboost import cross_validate as only_xgb_validate
from GS_Train_LinearR import cross_validate as lr_cross_validate
from GS_Train_Only_LR import cross_validate as only_lr_validate
import json

def run_xgb_cross_validate(sampled_vehicle_files, selected_car, adjusted_params_XGB, results_dict, size):
    try:
        xgb_results, xgb_scaler, _ = xgb_cross_validate(sampled_vehicle_files, selected_car, params=adjusted_params_XGB, plot=False, save_dir=None)
        if xgb_results:
            rmse_values = [xgb_result['rmse'] for xgb_result in xgb_results]
            results_dict[selected_car][size].append({
                'model': 'Hybrid Model(XGBoost)',
                'rmse': rmse_values
            })
    except Exception as e:
        print(f"XGBoost cross_validate error: {e}")

def run_only_xgb_validate(sampled_vehicle_files, selected_car, adjusted_params_ML, results_dict, size):
    try:
        only_xgb_results, only_xgb_scaler, _ = only_xgb_validate(sampled_vehicle_files, selected_car, params=adjusted_params_ML, plot=False)
        if only_xgb_results:
            rmse_values = [only_xgb_result['rmse'] for only_xgb_result in only_xgb_results]
            results_dict[selected_car][size].append({
                'model': 'Only ML(XGBoost)',
                'rmse': rmse_values
            })
    except Exception as e:
        print(f"Only ML(XGBoost) cross_validate error: {e}")

def run_evaluate(vehicle_files, selected_car):
    vehicle_file_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000]

    l2lambda = {selected_car: []}
    results_dict = {selected_car: {}}
    max_samples = len(vehicle_files[selected_car])

    # Filter sizes that are less than or equal to max_samples
    filtered_vehicle_file_sizes = [size for size in vehicle_file_sizes if size <= max_samples]

    # If max_samples is not in vehicle_file_sizes, add it
    if max_samples not in filtered_vehicle_file_sizes:
        filtered_vehicle_file_sizes.append(max_samples)

    # Sort the list to maintain order
    filtered_vehicle_file_sizes = sorted(filtered_vehicle_file_sizes)

    # Obtain the best hyperparameters using the full dataset
    _, _, best_params_XGB = xgb_cross_validate(vehicle_files, selected_car, params=None, plot=False, save_dir=None)

    l2lambda[selected_car].append({
        'model': 'Hybrid Model(XGBoost)',
        'params': best_params_XGB
    })

    _, _, best_params_ML = only_xgb_validate(vehicle_files, selected_car, params=None, plot=False)
    l2lambda[selected_car].append({
        'model': 'Only ML(XGBoost)',
        'params': best_params_ML
    })

    N_total = max_samples

    for size in filtered_vehicle_file_sizes:
        if size not in results_dict[selected_car]:
            results_dict[selected_car][size] = []
        if size < 20:
            samplings = 200
        elif 20 <= size < 50:
            samplings = 10
        elif 50 <= size <= 100:
            samplings = 5
        else:
            samplings = 1

        for sampling in range(samplings):
            sampled_files = random.sample(vehicle_files[selected_car], size)
            sampled_vehicle_files = {selected_car: sampled_files}

            # Compute RMSE for the Physics-based model
            rmse_physics = compute_rmse(sampled_vehicle_files, selected_car)
            if rmse_physics is not None:
                results_dict[selected_car][size].append({
                    'model': 'Physics-Based',
                    'rmse': [rmse_physics]
                })

            # Adjust hyperparameters for models that require them
            adjustment_factor = N_total / size
            adjusted_params_XGB = best_params_XGB.copy()
            for param in adjusted_params_XGB:
                if param != 'eta':  # Exclude learning_rate
                    adjusted_params_XGB[param] *= adjustment_factor

            adjusted_params_ML = best_params_ML.copy()
            for param in adjusted_params_ML:
                if param != 'eta':  # Exclude learning_rate
                    adjusted_params_ML[param] *= adjustment_factor

            # Create threads
            xgb_thread = Thread(target=run_xgb_cross_validate, args=(sampled_vehicle_files, selected_car, adjusted_params_XGB, results_dict, size))
            ml_thread = Thread(target=run_only_xgb_validate, args=(sampled_vehicle_files, selected_car, adjusted_params_ML, results_dict, size))

            # Start threads
            xgb_thread.start()
            ml_thread.start()

            # Wait for threads to complete
            xgb_thread.join()
            ml_thread.join()

            # Hybrid Model (Linear Regression) execution
            try:
                hybrid_lr_results, hybrid_lr_scaler = lr_cross_validate(sampled_vehicle_files, selected_car)
                if hybrid_lr_results:
                    hybrid_lr_rmse_values = [result['rmse'] for result in hybrid_lr_results]
                    results_dict[selected_car][size].append({
                        'model': 'Hybrid Model(Linear Regression)',
                        'rmse': hybrid_lr_rmse_values
                    })
            except Exception as e:
                print(f"Linear Regression cross_validate error: {e}")

            # Only LR execution
            try:
                only_lr_results, only_lr_scaler = only_lr_validate(sampled_vehicle_files, selected_car)
                if only_lr_results:
                    only_lr_rmse_values = [result['rmse'] for result in only_lr_results]
                    results_dict[selected_car][size].append({
                        'model': 'Only ML(LR)',
                        'rmse': only_lr_rmse_values
                    })
            except Exception as e:
                print(f"Only LR cross_validate error: {e}")
    print(results_dict)
    print("---------------------------\n")
    print(l2lambda)

    # 결과를 파일로 저장합니다.
    output_file_name = f"{selected_car}_results.txt"
    output_file_path = os.path.join("C:\\Users\\BSL\\Desktop", output_file_name)
    with open(output_file_path, 'w') as outfile:
        json.dump(results_dict, outfile)

    return results_dict

def plot_rmse_results(results_dict, selected_car, save_path):
    results_car = results_dict[selected_car]
    # Convert the keys of results_car to integers
    results_car = {int(size): data for size, data in results_car.items()}
    sizes = sorted(results_car.keys())
    sizes = [s for s in sizes if s >= 10]

    # 평균 RMSE와 표준편차를 저장할 리스트 초기화
    phys_rmse_mean = []
    phys_rmse_std = []
    xgb_rmse_mean = []
    xgb_rmse_std = []
    lr_rmse_mean = []
    lr_rmse_std = []
    only_ml_rmse_mean = []
    only_ml_rmse_std = []
    only_lr_rmse_mean = []
    only_lr_rmse_std = []

    for size in sizes:
        # 각 모델별 RMSE 값 추출
        phys_values = [item for result in results_car[size] if result['model'] == 'Physics-Based'
                       for item in result['rmse']]
        xgb_values = [item for result in results_car[size] if result['model'] == 'Hybrid Model(XGBoost)'
                      for item in result['rmse']]
        lr_values = [item for result in results_car[size] if result['model'] == 'Hybrid Model(Linear Regression)'
                     for item in result['rmse']]
        only_ml_values = [item for result in results_car[size] if
                          result['model'] == 'Only ML(XGBoost)' for
                          item in result['rmse']]
        only_lr_values = [item for result in results_car[size] if
                          result['model'] == 'Only ML(LR)' for
                          item in result['rmse']]

        # 각 모델별 평균과 표준편차 계산
        if phys_values:
            phys_mean = np.mean(phys_values)
            phys_std = np.std(phys_values)
        else:
            phys_mean = 1.0  # 기본값
            phys_std = 0.0
        phys_rmse_mean.append(phys_mean)
        phys_rmse_std.append(phys_std)

        if xgb_values:
            xgb_mean = np.mean(xgb_values)
            xgb_std = np.std(xgb_values)
            xgb_rmse_mean.append(xgb_mean)
            xgb_rmse_std.append(xgb_std)
        else:
            xgb_rmse_mean.append(None)
            xgb_rmse_std.append(None)

        if lr_values:
            lr_mean = np.mean(lr_values)
            lr_std = np.std(lr_values)
            lr_rmse_mean.append(lr_mean)
            lr_rmse_std.append(lr_std)
        else:
            lr_rmse_mean.append(None)
            lr_rmse_std.append(None)

        if only_ml_values:
            only_ml_mean = np.mean(only_ml_values)
            only_ml_std = np.std(only_ml_values)
            only_ml_rmse_mean.append(only_ml_mean)
            only_ml_rmse_std.append(only_ml_std)
        else:
            only_ml_rmse_mean.append(None)
            only_ml_rmse_std.append(None)

        if only_lr_values:
            only_lr_mean = np.mean(only_lr_values)
            only_lr_std = np.std(only_lr_values)
            only_lr_rmse_mean.append(only_lr_mean)
            only_lr_rmse_std.append(only_lr_std)
        else:
            only_lr_rmse_mean.append(None)
            only_lr_rmse_std.append(None)

    # 정규화된 RMSE 계산 (Physics-Based 모델의 RMSE를 1로 설정)
    normalized_xgb_rmse_mean = [x / p if p != 0 else 0 for x, p in zip(xgb_rmse_mean, phys_rmse_mean)]
    normalized_xgb_rmse_std = [ (x / p if p != 0 else 0) * 0.6745 for x, p in zip(xgb_rmse_std, phys_rmse_mean)]
    normalized_lr_rmse_mean = [x / p if p != 0 else 0 for x, p in zip(lr_rmse_mean, phys_rmse_mean)]
    normalized_lr_rmse_std = [ (x / p if p != 0 else 0) * 0.6745 for x, p in zip(lr_rmse_std, phys_rmse_mean)]
    normalized_only_ml_rmse_mean = [x / p if p != 0 else 0 for x, p in zip(only_ml_rmse_mean, phys_rmse_mean)]
    normalized_only_ml_rmse_std = [ (x / p if p != 0 else 0) * 0.6745 for x, p in zip(only_ml_rmse_std, phys_rmse_mean)]
    normalized_only_lr_rmse_mean = [x / p if p != 0 else 0 for x, p in zip(only_lr_rmse_mean, phys_rmse_mean)]
    normalized_only_lr_rmse_std = [ (x / p if p != 0 else 0) * 0.6745 for x, p in zip(only_lr_rmse_std, phys_rmse_mean)]
    normalized_phys_rmse_mean = [1.0 for _ in phys_rmse_mean]
    normalized_phys_rmse_std = [0.0 for _ in phys_rmse_mean]  # 항상 1이므로 표준편차 없음

    # # 정규화된 RMSE 플롯
    # plt.figure(figsize=(6, 5))
    # plt.errorbar(sizes, normalized_phys_rmse_mean, yerr=normalized_phys_rmse_std, label='Physics-Based', linestyle='--', color='#FF6347', capsize=5)
    # plt.errorbar(sizes, normalized_only_ml_rmse_mean, yerr=normalized_only_ml_rmse_std, label='Only ML(XGBoost)', marker='o', color='#32CD32', mfc='none',capsize=5)
    # plt.errorbar(sizes, normalized_only_lr_rmse_mean, yerr=normalized_only_lr_rmse_std, label='Only ML(LR)', marker='o', color='#FFA500', mfc='none', capsize=5)
    # plt.errorbar(sizes, normalized_lr_rmse_mean, yerr=normalized_lr_rmse_std, label='Hybrid Model(Linear Regression)', marker='o', color='#4682B4', mfc='none',capsize=5)
    # plt.errorbar(sizes, normalized_xgb_rmse_mean, yerr=normalized_xgb_rmse_std, label='Hybrid Model(XGBoost)', marker='D', color='#FFD700', mfc='none', capsize=5)

    # 정규화된 RMSE 플롯
    plt.figure(figsize=(6, 5))
    plt.plot(sizes, normalized_phys_rmse_mean, label='Physics-Based', linestyle='--', color='#FF6347')
    plt.plot(sizes, normalized_only_ml_rmse_mean, label='Only ML(XGBoost)', marker='o', color='#32CD32')
    plt.plot(sizes, normalized_only_lr_rmse_mean, label='Only ML(LR)', marker='o', color='#FFA500')
    plt.plot(sizes, normalized_lr_rmse_mean, label='Hybrid Model(LR)', marker='o', color='#4682B4')
    plt.plot(sizes, normalized_xgb_rmse_mean, label='Hybrid Model(XGBoost)', marker='D', color='#FFD700')

    plt.xlabel('Number of Trips')
    plt.ylabel('Normalized RMSE')
    plt.title(f'RMSE vs Number of Trips for {selected_car}')
    plt.legend()
    plt.grid(False)
    plt.xscale('log')
    plt.xticks(sizes, [str(size) for size in sizes], rotation=45)
    plt.xlim(min(sizes) - 1, max(sizes) - 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{selected_car}_rmse_normalized.png"), dpi=300)
    plt.show()
