import os
import numpy as np
import random
from GS_Functions import get_vehicle_files, compute_mape_rrmse
from GS_Train_XGboost import cross_validate as xgb_cross_validate, add_predicted_power_column as xgb_add_predicted_power_column
from GS_Train_Only_XGboost import cross_validate as only_xgb_validate, add_predicted_power_column as only_xgb_add_predicted_power_column
from GS_Train_LinearR import cross_validate as lr_cross_validate
from GS_Train_LightGBM import cross_validate as lgbm_cross_validate
import matplotlib as plt
def run_evaluate(vehicle_files, selected_car):
    vehicle_file_sizes = [5, 7, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000]

    l2lambda = {selected_car: []}
    results_dict = {selected_car: {}}
    max_samples = len(vehicle_files[selected_car])

    filtered_vehicle_file_sizes = [size for size in vehicle_file_sizes if size <= max_samples]
    _, _, lambda_XGB = xgb_cross_validate(vehicle_files, selected_car, None, None, save_dir=None)
    l2lambda[selected_car].append({
        'model': 'Hybrid Model(XGBoost)',
        'lambda': lambda_XGB
    })
    _, _, lambda_LGBM = lgbm_cross_validate(vehicle_files, selected_car, None, None, save_dir=None)
    l2lambda[selected_car].append({
        'model': 'Hybrid Model(LGBM)',
        'lambda': lambda_LGBM
    })
    _, _, lambda_ML = only_xgb_validate(vehicle_files, selected_car, None, None, save_dir=None)
    l2lambda[selected_car].append({
        'model': 'Only ML(XGBoost)',
        'lambda': lambda_ML
    })

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

            # Physics-based model RRMSE calculation
            mape_physics, rrmse_physics = compute_mape_rrmse(sampled_vehicle_files, selected_car)
            if rrmse_physics is not None:
                results_dict[selected_car][size].append({
                    'model': 'Physics-Based',
                    'mape': [mape_physics],
                    'rrmse': [rrmse_physics]
                })

            results, scaler, _ = xgb_cross_validate(sampled_vehicle_files, selected_car, lambda_XGB, None, save_dir=None)
            if results:
                mape_values = [mape for _, _, _, _, _, mape in results]
                rrmse_values = [rrmse for _, _, _, _, rrmse, _ in results]
                results_dict[selected_car][size].append({
                    'model': 'Hybrid Model(XGBoost)',
                    'mape': mape_values,
                    'rrmse': rrmse_values
                })

            results, scaler = lr_cross_validate(sampled_vehicle_files, selected_car, None, save_dir=None)
            if results:
                mape_values = [mape for _, _, _, _, _, mape in results]
                rrmse_values = [rrmse for _, _, _, _, rrmse, _ in results]
                results_dict[selected_car][size].append({
                    'model': 'Hybrid Model(Linear Regression)',
                    'mape': mape_values,
                    'rrmse': rrmse_values
                })

            results, scaler, _ = only_xgb_validate(sampled_vehicle_files, selected_car, lambda_ML, None, save_dir=None)
            if results:
                mape_values = [mape for _, _, _, _, _, mape in results]
                rrmse_values = [rrmse for _, _, _, _, rrmse, _ in results]
                results_dict[selected_car][size].append({
                    'model': 'Only ML(XGBoost)',
                    'mape': mape_values,
                    'rrmse': rrmse_values
                })

    print(results_dict)
    return results_dict

def plot_mape_results(results_dict, selected_car, save_path):
    results_car = results_dict[selected_car]
    sizes = sorted(results_car.keys())

    phys_rrmse = []
    phys_std = []
    xgb_rrmse = []
    xgb_std = []
    lr_rrmse = []
    lr_std = []
    only_ml_rrmse = []
    only_ml_std = []

    phys_mape = []
    phys_mape_std = []
    xgb_mape = []
    xgb_mape_std = []
    lr_mape = []
    lr_mape_std = []
    only_ml_mape = []
    only_ml_mape_std = []

    for size in sizes:
        phys_values = [item for result in results_car[size] if result['model'] == 'Physics-Based'
                       for item in result['rrmse']]
        xgb_values = [item for result in results_car[size] if result['model'] == 'Hybrid Model(XGBoost)'
                      for item in result['rrmse']]
        lr_values = [item for result in results_car[size] if result['model'] == 'Hybrid Model(Linear Regression)'
                     for item in result['rrmse']]
        only_ml_values = [item for result in results_car[size] if
                          result['model'] == 'Only ML(XGBoost)' for
                          item in result['rrmse']]
        phys_mape_values = [item for result in results_car[size] if
                            result['model'] == 'Physics-Based' for item in result['mape']]
        xgb_mape_values = [item for result in results_car[size] if
                           result['model'] == 'Hybrid Model(XGBoost)' for item in result['mape']]
        lr_mape_values = [item for result in results_car[size] if
                          result['model'] == 'Hybrid Model(Linear Regression)' for item in
                          result['mape']]
        only_ml_mape_values = [item for result in results_car[size] if
                               result['model'] == 'Only ML(XGBoost)' for item in result['mape']]

        if phys_values:
            phys_rrmse.append(np.mean(phys_values))
            phys_std.append(2 * np.std(phys_values))
        if xgb_values:
            xgb_rrmse.append(np.mean(xgb_values))
            xgb_std.append(2 * np.std(xgb_values))  # 2σ 95%
        if lr_values:
            lr_rrmse.append(np.mean(lr_values))
            lr_std.append(2 * np.std(lr_values))  # 2σ 95%
        if only_ml_values:
            only_ml_rrmse.append(np.mean(only_ml_values))
            only_ml_std.append(2 * np.std(only_ml_values))  # 2σ 95%

        # MAPE 수집
        if phys_mape_values:
            phys_mape.append(np.mean(phys_mape_values))
            phys_mape_std.append(2 * np.std(phys_mape_values))
        if xgb_mape_values:
            xgb_mape.append(np.mean(xgb_mape_values))
            xgb_mape_std.append(2 * np.std(xgb_mape_values))  # 2σ 95%
        if lr_mape_values:
            lr_mape.append(np.mean(lr_mape_values))
            lr_mape_std.append(2 * np.std(lr_mape_values))  # 2σ 95%
        if only_ml_mape_values:
            only_ml_mape.append(np.mean(only_ml_mape_values))
            only_ml_mape_std.append(2 * np.std(only_ml_mape_values))  # 2σ 95%

    # MAPE 플롯
    plt.figure(figsize=(10, 8))

    # Physics-Based Model (MAPE)
    plt.plot(sizes, phys_mape, label='Physics-Based', linestyle='--', color='r')

    # Hybrid Model(XGBoost) (MAPE)
    plt.errorbar(sizes, xgb_mape, yerr=xgb_mape_std, label='Hybrid Model(XGBoost)', marker='o',
                 capsize=5)

    # Hybrid Model(Linear Regression) (MAPE)
    plt.errorbar(sizes, lr_mape, yerr=lr_mape_std, label='Hybrid Model(Linear Regression)',
                 marker='o', capsize=5)

    # Only ML(XGBoost) (MAPE)
    plt.errorbar(sizes, only_ml_mape, yerr=only_ml_mape_std, label='Only ML(XGBoost)', marker='o',
                 capsize=5)

    plt.xlabel('Number of Trips')
    plt.ylabel('MAPE')
    plt.title(f'MAPE vs Number of Trips for {selected_car}')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.xticks(sizes, [str(size) for size in sizes], rotation=45)
    plt.xlim(min(sizes) - 1, max(sizes) + 1000)
    plt.savefig(os.path.join(save_path, f"{selected_car}_mape.png"), dpi=600)
    plt.show()

def plot_rrmse_results(results_dict, selected_car, save_path):
    results_car = results_dict[selected_car]
    sizes = sorted(results_car.keys())

    phys_rrmse = []
    phys_std = []
    xgb_rrmse = []
    xgb_std = []
    lr_rrmse = []
    lr_std = []
    only_ml_rrmse = []
    only_ml_std = []

    for size in sizes:
        phys_values = [item for result in results_car[size] if result['model'] == 'Physics-Based'
                       for item in result['rrmse']]
        xgb_values = [item for result in results_car[size] if result['model'] == 'Hybrid Model(XGBoost)'
                      for item in result['rrmse']]
        lr_values = [item for result in results_car[size] if result['model'] == 'Hybrid Model(Linear Regression)'
                     for item in result['rrmse']]
        only_ml_values = [item for result in results_car[size] if
                          result['model'] == 'Only ML(XGBoost)' for
                          item in result['rrmse']]

        if phys_values:
            phys_rrmse.append(np.mean(phys_values))
            phys_std.append(2 * np.std(phys_values))
        if xgb_values:
            xgb_rrmse.append(np.mean(xgb_values))
            xgb_std.append(2 * np.std(xgb_values))  # 2σ 95%
        if lr_values:
            lr_rrmse.append(np.mean(lr_values))
            lr_std.append(2 * np.std(lr_values))  # 2σ 95%
        if only_ml_values:
            only_ml_rrmse.append(np.mean(only_ml_values))
            only_ml_std.append(2 * np.std(only_ml_values))  # 2σ 95%

    # RRMSE 플롯
    plt.figure(figsize=(10, 8))

    # Physics-Based Model (RRMSE)
    plt.plot(sizes, phys_rrmse, label='Physics-Based', linestyle='--', color='r')

    # Hybrid Model(XGBoost) (RRMSE)
    plt.errorbar(sizes, xgb_rrmse, yerr=xgb_std, label='Hybrid Model(XGBoost)', marker='o',
                 capsize=5)

    # Hybrid Model(Linear Regression) (RRMSE)
    plt.errorbar(sizes, lr_rrmse, yerr=lr_std, label='Hybrid Model(Linear Regression)', marker='o',
                 capsize=5)

    # Only ML(XGBoost) (RRMSE)
    plt.errorbar(sizes, only_ml_rrmse, yerr=only_ml_std, label='Only ML(XGBoost)', marker='o',
                 capsize=5)

    plt.xlabel('Number of Trips')
    plt.ylabel('RRMSE')
    plt.title(f'RRMSE vs Number of Trips for {selected_car}')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.xticks(sizes, [str(size) for size in sizes], rotation=45)
    plt.xlim(min(sizes) - 1, max(sizes) + 1000)
    plt.savefig(os.path.join(save_path, f"{selected_car}_rrmse.png"), dpi=600)
    plt.show()