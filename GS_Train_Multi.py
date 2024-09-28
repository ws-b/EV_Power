import os
import numpy as np
import random
import matplotlib.pyplot as plt
from threading import Thread
from GS_Functions import get_vehicle_files, compute_mape_rrmse
from GS_Train_XGboost import cross_validate as xgb_cross_validate
from GS_Train_Only_XGboost import cross_validate as only_xgb_validate
from GS_Train_LinearR import cross_validate as lr_cross_validate
from GS_Train_LightGBM import cross_validate as lgbm_cross_validate

def run_xgb_cross_validate(sampled_vehicle_files, selected_car, adjusted_params_XGB, results_dict, size):
    try:
        results, scaler, _ = xgb_cross_validate(sampled_vehicle_files, selected_car, params=adjusted_params_XGB, plot=False, save_dir=None)
        if results:
            rmse_values = [result['rmse'] for result in results]
            results_dict[selected_car][size].append({
                'model': 'Hybrid Model(XGBoost)',
                'rmse': rmse_values
            })
    except Exception as e:
        print(f"XGBoost cross_validate error: {e}")

def run_lgbm_cross_validate(sampled_vehicle_files, selected_car, adjusted_params_LGBM, results_dict, size):
    try:
        results, scaler, _ = lgbm_cross_validate(sampled_vehicle_files, selected_car, params=adjusted_params_LGBM, plot=False, save_dir=None)
        if results:
            rmse_values = [result['rmse'] for result in results]
            results_dict[selected_car][size].append({
                'model': 'Hybrid Model(LightGBM)',
                'rmse': rmse_values
            })
    except Exception as e:
        print(f"LightGBM cross_validate error: {e}")

def run_evaluate(vehicle_files, selected_car):
    vehicle_file_sizes = [5, 7, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000]

    l2lambda = {selected_car: []}
    results_dict = {selected_car: {}}
    max_samples = len(vehicle_files[selected_car])

    filtered_vehicle_file_sizes = [size for size in vehicle_file_sizes if size <= max_samples]
    # 전체 데이터로 최적의 하이퍼파라미터 얻기
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

    _, _, best_params_LGBM = lgbm_cross_validate(vehicle_files, selected_car, params=None, plot=False)
    l2lambda[selected_car].append({
        'model': 'Hybrid Model(LightGBM)',
        'params': best_params_LGBM
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

            # Physics-based model RMSE 계산
            _, _, rmse_physics = compute_mape_rrmse(sampled_vehicle_files, selected_car)
            if rmse_physics is not None:
                results_dict[selected_car][size].append({
                    'model': 'Physics-Based',
                    'rmse': [rmse_physics]
                })

            # 하이퍼파라미터 조정
            adjustment_factor = N_total / size
            adjusted_params_XGB = best_params_XGB.copy()
            for param in adjusted_params_XGB:
                if param != 'eta':  # learning_rate 제외
                    adjusted_params_XGB[param] *= adjustment_factor

            adjusted_params_ML = best_params_ML.copy()
            for param in adjusted_params_ML:
                if param != 'eta':  # learning_rate 제외
                    adjusted_params_ML[param] *= adjustment_factor

            adjusted_params_LGBM = best_params_LGBM.copy()
            for param in adjusted_params_LGBM:
                if param != 'learning_rate':  # learning_rate 제외
                    adjusted_params_LGBM[param] *= adjustment_factor

            # 스레드 생성
            xgb_thread = Thread(target=run_xgb_cross_validate, args=(sampled_vehicle_files, selected_car, adjusted_params_XGB, results_dict, size))
            lgbm_thread = Thread(target=run_lgbm_cross_validate, args=(sampled_vehicle_files, selected_car, adjusted_params_LGBM, results_dict, size))

            # 스레드 시작
            xgb_thread.start()
            lgbm_thread.start()

            # 스레드 완료 대기
            xgb_thread.join()
            lgbm_thread.join()

            # Hybrid Model(Linear Regression) 실행 (직렬 실행)
            try:
                results, scaler, _ = lr_cross_validate(sampled_vehicle_files, selected_car, params=None)
                if results:
                    rmse_values = [result['rmse'] for result in results]
                    results_dict[selected_car][size].append({
                        'model': 'Hybrid Model(Linear Regression)',
                        'rmse': rmse_values
                    })
            except Exception as e:
                print(f"Linear Regression cross_validate error: {e}")

            # Only ML(XGBoost) 실행 (직렬 실행)
            try:
                results, scaler, _ = only_xgb_validate(sampled_vehicle_files, selected_car, params=adjusted_params_ML, plot=False)
                if results:
                    rmse_values = [result['rmse'] for result in results]
                    results_dict[selected_car][size].append({
                        'model': 'Only ML(XGBoost)',
                        'rmse': rmse_values
                    })
            except Exception as e:
                print(f"Only ML(XGBoost) cross_validate error: {e}")

    print(results_dict)
    return results_dict

def plot_rmse_results(results_dict, selected_car, save_path):
    results_car = results_dict[selected_car]
    sizes = sorted(results_car.keys())

    phys_rmse = []
    xgb_rmse = []
    lr_rmse = []
    only_ml_rmse = []
    lgbm_rmse = []

    for size in sizes:
        phys_values = [item for result in results_car[size] if result['model'] == 'Physics-Based'
                       for item in result['rmse']]
        xgb_values = [item for result in results_car[size] if result['model'] == 'Hybrid Model(XGBoost)'
                      for item in result['rmse']]
        lr_values = [item for result in results_car[size] if result['model'] == 'Hybrid Model(Linear Regression)'
                     for item in result['rmse']]
        only_ml_values = [item for result in results_car[size] if
                          result['model'] == 'Only ML(XGBoost)' for
                          item in result['rmse']]
        lgbm_values = [item for result in results_car[size] if
                       result['model'] == 'Hybrid Model(LightGBM)' for
                       item in result['rmse']]

        if phys_values:
            phys_mean_rmse = np.mean(phys_values)
            phys_rmse.append(phys_mean_rmse)
        else:
            phys_mean_rmse = 1.0  # 기본값
            phys_rmse.append(phys_mean_rmse)

        if xgb_values:
            xgb_mean_rmse = np.mean(xgb_values)
            xgb_rmse.append(xgb_mean_rmse)

        if lr_values:
            lr_mean_rmse = np.mean(lr_values)
            lr_rmse.append(lr_mean_rmse)

        if only_ml_values:
            only_ml_mean_rmse = np.mean(only_ml_values)
            only_ml_rmse.append(only_ml_mean_rmse)

        if lgbm_values:
            lgbm_mean_rmse = np.mean(lgbm_values)
            lgbm_rmse.append(lgbm_mean_rmse)

    # 정규화된 RMSE 계산 (Physics-Based 모델의 RMSE를 1로 설정)
    normalized_xgb_rmse = [x / p if p != 0 else 0 for x, p in zip(xgb_rmse, phys_rmse)]
    normalized_lr_rmse = [x / p if p != 0 else 0 for x, p in zip(lr_rmse, phys_rmse)]
    normalized_only_ml_rmse = [x / p if p != 0 else 0 for x, p in zip(only_ml_rmse, phys_rmse)]
    normalized_lgbm_rmse = [x / p if p != 0 else 0 for x, p in zip(lgbm_rmse, phys_rmse)]
    normalized_phys_rmse = [p / p if p != 0 else 0 for p in phys_rmse]  # 항상 1

    # 비정규화된 RMSE 플롯
    plt.figure(figsize=(10, 8))
    plt.plot(sizes, phys_rmse, label='Physics-Based', linestyle='--', color='r')
    plt.plot(sizes, xgb_rmse, label='Hybrid Model(XGBoost)', marker='o')
    plt.plot(sizes, lr_rmse, label='Hybrid Model(Linear Regression)', marker='o')
    plt.plot(sizes, only_ml_rmse, label='Only ML(XGBoost)', marker='o')
    plt.plot(sizes, lgbm_rmse, label='Hybrid Model(LightGBM)', marker='o')
    plt.xlabel('Number of Trips')
    plt.ylabel('RMSE')
    plt.title(f'RMSE vs Number of Trips for {selected_car} (Unnormalized)')
    plt.legend()
    plt.grid(False)
    plt.xscale('log')
    plt.xticks(sizes, [str(size) for size in sizes], rotation=45)
    plt.xlim(min(sizes) - 1, max(sizes) + 1000)
    plt.savefig(os.path.join(save_path, f"{selected_car}_rmse_unnormalized.png"), dpi=600)
    plt.show()

    # 정규화된 RMSE 플롯
    plt.figure(figsize=(10, 8))
    plt.plot(sizes, normalized_phys_rmse, label='Physics-Based (Normalized)', linestyle='--', color='r')
    plt.plot(sizes, normalized_xgb_rmse, label='Hybrid Model(XGBoost) (Normalized)', marker='o')
    plt.plot(sizes, normalized_lr_rmse, label='Hybrid Model(Linear Regression) (Normalized)', marker='o')
    plt.plot(sizes, normalized_only_ml_rmse, label='Only ML(XGBoost) (Normalized)', marker='o')
    plt.plot(sizes, normalized_lgbm_rmse, label='Hybrid Model(LightGBM) (Normalized)', marker='o')
    plt.xlabel('Number of Trips')
    plt.ylabel('Normalized RMSE')
    plt.title(f'Normalized RMSE vs Number of Trips for {selected_car}')
    plt.legend()
    plt.grid(False)
    plt.xscale('log')
    plt.xticks(sizes, [str(size) for size in sizes], rotation=45)
    plt.xlim(min(sizes) - 1, max(sizes) + 1000)
    plt.savefig(os.path.join(save_path, f"{selected_car}_rmse_normalized.png"), dpi=600)
    plt.show()
