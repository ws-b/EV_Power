import os
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from GS_Merge_Power import process_files_power, select_vehicle
from GS_Functions import get_vehicle_files, compute_rrmse, compute_rmse, compute_mape
from GS_plot import plot_power, plot_energy, plot_energy_scatter, plot_power_scatter, plot_energy_dis, plot_driver_energy_scatter, plot_contour2, plot_2d_histogram
from GS_vehicle_dict import vehicle_dict
from GS_Train_XGboost import cross_validate as xgb_cross_validate, add_predicted_power_column as xgb_add_predicted_power_column
from GS_Train_Only_XGboost import cross_validate as only_xgb_validate, add_predicted_power_column as only_xgb_add_predicted_power_column
from GS_Train_LinearR import cross_validate as lr_cross_validate, add_predicted_power_column as lr_add_predicted_power_column
from GS_Train_LightGBM import cross_validate as lgbm_cross_validate, add_predicted_power_column as lgbm_add_predicted_power_column

def main():
    car_options = {
        1: 'EV6',
        2: 'Ioniq5',
        3: 'KonaEV',
        4: 'NiroEV',
        5: 'GV60',
        6: 'Ioniq6',
        7: 'Bongo3EV',
        8: 'Porter2EV',
    }

    folder_path = os.path.join(os.path.normpath(r'D:\SamsungSTF\Processed_Data'), 'TripByTrip')

    while True:
        print("1: Calculate Power(W)")
        print("2: Train Model")
        print("3: Predicted Power(W) using Trained Model")
        print("4: Plotting Graph (Power & Energy)")
        print("5: Plotting Graph (Scatter, Energy Distribution)")
        print("6: Plotting Graph (Speed, Acceleration, etc.)")
        print("0: Quitting the program.")
        try:
            task_choice = int(input("Enter number you want to run: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if task_choice in [1, 2, 3, 4, 5, 6]:
            selected_cars, vehicle_files = get_vehicle_files(car_options, folder_path, vehicle_dict)
            if not selected_cars:
                print("No cars selected. Returning to main menu.")
                continue

        if task_choice == 1:
            for selected_car in selected_cars:
                EV = select_vehicle(selected_car)
                filtered_files = [f for f in vehicle_files.get(selected_car, []) if f.endswith('.csv') and 'bms' in f and 'altitude' in f]
                unfiltered_files = vehicle_files.get(selected_car, [])
                process_files_power(unfiltered_files, EV)

        elif task_choice == 2:
            save_dir = os.path.join(os.path.dirname(folder_path), 'Models')
            while True:
                print("1: Hybrid(XGB) Model")
                print("2: Hybrid(LR) Model")
                print("3: Only ML(XGB) Model")
                print("4: Hybrid(LGBM) Model")
                print("5: Train Models")
                print("6: Return to previous menu")
                print("0: Quitting the program")
                try:
                    train_choice = int(input("Enter number you want to run: "))
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

                if train_choice == 6:
                    break
                elif train_choice == 0:
                    print("Quitting the program.")
                    return
                XGB = {}
                LR = {}
                LGBM = {}
                ONLY_ML = {}
                
                for selected_car in selected_cars:
                    if train_choice == 1:
                        results, scaler, _ = xgb_cross_validate(vehicle_files, selected_car, None, True,  save_dir=save_dir)

                        if results:
                            mape_values = []
                            rmse_values = []
                            rrmse_values = []
                            for fold_num, rrmse, rmse, mape in results:
                                print(f"Fold: {fold_num}, RRMSE: {rrmse}, MAPE: {mape}")
                                mape_values.append(mape)
                                rmse_values.append(rmse)
                                rrmse_values.append(rrmse)
                            XGB[selected_car] = {
                                'RMSE': rmse_values,
                                'RRMSE': rrmse_values,
                                'MAPE': mape_values
                            }

                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 2:
                        results, scaler = lr_cross_validate(vehicle_files, selected_car, True, save_dir=save_dir)

                        if results:
                            mape_values = []
                            rmse_values = []
                            rrmse_values = []
                            for fold_num, rrmse, rmse, mape in results:
                                print(f"Fold: {fold_num}, RRMSE: {rrmse}, MAPE: {mape}")
                                mape_values.append(mape)
                                rmse_values.append(rmse)
                                rrmse_values.append(rrmse)
                            LR[selected_car] = {
                                'RMSE': rmse_values,
                                'RRMSE': rrmse_values,
                                'MAPE': mape_values
                            }
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 3:
                        results, scaler, _ = only_xgb_validate(vehicle_files, selected_car, None, True, save_dir=save_dir)

                        if results:
                            mape_values = []
                            rmse_values = []
                            rrmse_values = []
                            for fold_num, rrmse, rmse, mape in results:
                                print(f"Fold: {fold_num}, RRMSE: {rrmse}, MAPE: {mape}")
                                mape_values.append(mape)
                                rmse_values.append(rmse)
                                rrmse_values.append(rrmse)
                            ONLY_ML[selected_car] = {
                                'RMSE': rmse_values,
                                'RRMSE': rrmse_values,
                                'MAPE': mape_values
                            }
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 4:
                        results, scaler, _ = lgbm_cross_validate(vehicle_files, selected_car, None, True,  save_dir=save_dir)

                        if results:
                            mape_values = []
                            rmse_values = []
                            rrmse_values = []
                            for fold_num, rrmse, rmse, mape in results:
                                print(f"Fold: {fold_num}, RRMSE: {rrmse}, MAPE: {mape}")
                                mape_values.append(mape)
                                rmse_values.append(rmse)
                                rrmse_values.append(rrmse)
                            LGBM[selected_car] = {
                                'RMSE': rmse_values,
                                'RRMSE': rrmse_values,
                                'MAPE': mape_values
                            }

                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 5:
                        vehicle_file_sizes = [5, 7, 10, 20, 50, 100, 200, 500,
                                              1000, 2000, 3000, 5000, 10000]

                        l2lambda = {selected_car: []}
                        results_dict = {selected_car: {}}
                        max_samples = len(vehicle_files[selected_car])

                        filtered_vehicle_file_sizes = [size for size in vehicle_file_sizes if size <= max_samples]
                        _, _, lambda_XGB = xgb_cross_validate(vehicle_files, selected_car, None, None, save_dir=None)
                        l2lambda[selected_car].append({
                                        'model': 'Hybrid Model(XGBoost)',
                                        'lambda': lambda_XGB
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
                                mape_physics = compute_mape(sampled_vehicle_files, selected_car)
                                rrmse_physics = compute_rrmse(sampled_vehicle_files, selected_car)
                                if rrmse_physics is not None:
                                    results_dict[selected_car][size].append({
                                        'model': 'Physics-Based',
                                        'rrmse': [rrmse_physics],
                                        'mape' : [mape_physics]
                                    })

                                results, scaler, _ = xgb_cross_validate(sampled_vehicle_files, selected_car, lambda_XGB, None, save_dir=None)
                                if results:
                                    mape_values = [mape for _, _, _, mape in results]
                                    rrmse_values = [rrmse for _, rrmse, _, _ in results]
                                    results_dict[selected_car][size].append({
                                        'model': 'Hybrid Model(XGBoost)',
                                        'rrmse': rrmse_values,
                                        'mape' : mape_values
                                    })

                                results, scaler = lr_cross_validate(sampled_vehicle_files, selected_car, None, save_dir=None)
                                if results:
                                    mape_values = [mape for _, _, _, mape in results]
                                    rrmse_values = [rrmse for _, rrmse, _, _ in results]
                                    results_dict[selected_car][size].append({
                                        'model': 'Hybrid Model(Linear Regression)',
                                        'rrmse': rrmse_values,
                                        'mape' : mape_values
                                    })

                                results, scaler, _ = only_xgb_validate(sampled_vehicle_files, selected_car, lambda_ML, None, save_dir=None)
                                if results:
                                    mape_values = [mape for _, _, _, mape in results]
                                    rrmse_values = [rrmse for _, rrmse, _, _ in results]
                                    results_dict[selected_car][size].append({
                                        'model': 'Only ML(XGBoost)',
                                        'rrmse': rrmse_values,
                                        'mape' : mape_values
                                    })

                        print(results_dict)

                        results_car = results_dict[selected_car]
                        sizes = sorted(results_car.keys())

                        phys_rrmse = []
                        xgb_rrmse = []
                        xgb_std = []
                        lr_rrmse = []
                        lr_std = []
                        only_ml_rrmse = []
                        only_ml_std = []

                        phys_mape = []
                        xgb_mape = []
                        xgb_mape_std = []
                        lr_mape = []
                        lr_mape_std = []
                        only_ml_mape = []
                        only_ml_mape_std = []

                        for size in sizes:
                            phys_values = [item for result in results_car[size] if result['model'] == 'Physics-Based' for item
                                           in result['rrmse']]
                            xgb_values = [item for result in results_car[size] if result['model'] == 'Hybrid Model(XGBoost)' for
                                          item in result['rrmse']]
                            lr_values = [item for result in results_car[size] if
                                         result['model'] == 'Hybrid Model(Linear Regression)' for item in result['rrmse']]
                            only_ml_values = [item for result in results_car[size] if result['model'] == 'Only ML(XGBoost)' for
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
                            if xgb_mape_values:
                                xgb_mape.append(np.mean(xgb_mape_values))
                                xgb_mape_std.append(2 * np.std(xgb_mape_values))  # 2σ 95%
                            if lr_mape_values:
                                lr_mape.append(np.mean(lr_mape_values))
                                lr_mape_std.append(2 * np.std(lr_mape_values))  # 2σ 95%
                            if only_ml_mape_values:
                                only_ml_mape.append(np.mean(only_ml_mape_values))
                                only_ml_mape_std.append(2 * np.std(only_ml_mape_values))  # 2σ 95%

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
                        plt.show()

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
                        plt.show()
                        print(f"{l2lambda}")

                print(f"XGB RRMSE & RMSE: {XGB}")
                print(f"LR RRMSE & RMSE: {LR}")
                print(f"ONLY ML RRMSE & RMSE: {ONLY_ML}")

        elif task_choice == 3:
            while True:
                print("1: XGBoost Model")
                print("2: Linear Regression")
                print("3: Machine Learning Only(XGB)")
                print("4: LightGBM Model")
                print("5: Return to previous menu")
                print("0: Quitting the program")
                try:
                    pred_choice = int(input("Enter number you want to run: "))
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

                if pred_choice == 5:
                    break
                elif pred_choice == 0:
                    print("Quitting the program")
                    return

                for selected_car in selected_cars:
                    if pred_choice == 1:
                        model_path = os.path.join(os.path.dirname(folder_path), 'Models', f'XGB_best_model_{selected_car}.json')
                        scaler_path = os.path.join(os.path.dirname(folder_path), 'Models', f'XGB_scaler_{selected_car}.pkl')

                        if not vehicle_files[selected_car]:
                            print(f"No files to process for the selected vehicle: {selected_car}")
                            continue

                        # Load the scaler
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)

                        xgb_add_predicted_power_column(vehicle_files[selected_car], model_path, scaler)
                    elif pred_choice == 2:
                        model_path = os.path.join(os.path.dirname(folder_path), 'Models', f'LR_best_model_{selected_car}.joblib')
                        scaler_path = os.path.join(os.path.dirname(folder_path), 'Models', f'LR_scaler_{selected_car}.pkl')

                        if not vehicle_files[selected_car]:
                            print(f"No files to process for the selected vehicle: {selected_car}")
                            continue

                        # Load the scaler
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)

                        lr_add_predicted_power_column(vehicle_files[selected_car], model_path, scaler)
                    elif pred_choice == 3:
                        model_path = os.path.join(os.path.dirname(folder_path), 'Models', f'XGB_Only_best_model_{selected_car}.json')
                        scaler_path = os.path.join(os.path.dirname(folder_path), 'Models', f'XGB_Only_scaler_{selected_car}.pkl')

                        if not vehicle_files[selected_car]:
                            print(f"No files to process for the selected vehicle: {selected_car}")
                            continue

                        # Load the scaler
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)

                        only_xgb_add_predicted_power_column(vehicle_files[selected_car], model_path, scaler)
                    elif pred_choice == 4:
                        model_path = os.path.join(os.path.dirname(folder_path), 'Models', f'XGB_best_model_{selected_car}.json')
                        scaler_path = os.path.join(os.path.dirname(folder_path), 'Models', f'XGB_scaler_{selected_car}.pkl')

                        if not vehicle_files[selected_car]:
                            print(f"No files to process for the selected vehicle: {selected_car}")
                            continue

                        # Load the scaler
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)

                        lgbm_add_predicted_power_column(vehicle_files[selected_car], model_path, scaler)

        elif task_choice == 4:
            while True:
                print("1: Plotting Stacked Power Plot Term by Term")
                print("2: Plotting Model's Power Graph")
                print("3: Plotting Data's Power Graph")
                print("4: Plotting Data & Physics Model Power Comparison Graph ")
                print("5: Plotting Comparison Graph(Phys, Data, Hybrid)")
                print("6: Plotting Power & Altitude Graph")
                print("7: Plotting Model's Energy Graph")
                print("8: Plotting Data's Energy Graph")
                print("9: Plotting Energy Comparison Graph")
                print("10: Plotting Altitude and Energy Graph")
                print("13: Return to previous menu")
                print("0: Quitting the program.")

                selections = input("Enter the numbers you want to run, separated by commas (e.g., 1,2,3): ")
                selections_list = selections.split(',')

                for selection in selections_list:
                    try:
                        plot = int(selection.strip())
                    except ValueError:
                        print(f"Invalid input: {selection}. Please enter a valid number.")
                        continue

                    for selected_car in selected_cars:
                        sample_files = random.sample(vehicle_files[selected_car] ,5)
                        altitude_files = [file for file in vehicle_files[selected_car] if 'altitude' in file]
                        sample_alt_files = random.sample(altitude_files, min(5, len(altitude_files)))
                        if plot == 1:
                            plot_power(sample_files, selected_car, 'stacked')
                        elif plot == 2:
                            plot_power(sample_files, selected_car, 'physics')
                        elif plot == 3:
                            plot_power(sample_files, selected_car, 'data')
                        elif plot == 4:
                            plot_power(sample_files, selected_car, 'comparison')
                        elif plot == 5:
                            plot_power(sample_files, selected_car, 'hybrid')
                        elif plot == 6:
                            plot_power(sample_alt_files, selected_car, 'altitude')
                        elif plot == 7:
                            plot_energy(sample_files, selected_car, 'physics')
                        elif plot == 8:
                            plot_energy(sample_files, selected_car, 'data')
                        elif plot == 9:
                            plot_energy(sample_files, selected_car, 'comparison')
                        elif plot == 10:
                            plot_energy(sample_alt_files, selected_car, 'altitude')
                        elif plot == 13:
                            break
                        elif plot == 0:
                            print("Quitting the program.")
                            return
                        else:
                            print(f"Invalid choice: {plot}. Please try again.")
                else:
                    continue
                break

        elif task_choice == 5:
            while True:
                print("1: Plotting Energy Scatter Graph")
                print("2: Plotting Learning Scatter Graph")
                print("3: Plotting Individual Driver's Scatter Graph")
                print("4: Plotting Power and Delta_altitude Graph")
                print("5: Plotting Physics Model Energy Distribution Graph")
                print("6: Plotting Data Energy Distribution Graph")
                print("7: Plotting Learning Energy Distribution Graph")
                print("9: Return to previous menu.")
                print("0: Quitting the program.")

                selections = input("Enter the numbers you want to run, separated by commas (e.g., 1,2,3): ")
                selections_list = selections.split(',')
                for selection in selections_list:
                    try:
                        plot = int(selection.strip())
                    except ValueError:
                        print(f"Invalid input: {selection}. Please enter a valid number.")
                        continue
                    if plot == 9:
                        break
                    elif plot == 0:
                        print("Quitting the program")
                        return
                    for selected_car in selected_cars:
                        if plot == 1:
                            plot_energy_scatter(vehicle_files[selected_car], selected_car, 'physics')
                        elif plot == 2:
                            plot_energy_scatter(vehicle_files[selected_car], selected_car, 'hybrid')
                        elif plot == 3:
                            if len(vehicle_dict[selected_car]) >=5 :
                                sample_ids = random.sample(vehicle_dict[selected_car], 5)
                                sample_files_dict = {id: [f for f in vehicle_files[selected_car] if id in f] for id in sample_ids}
                                plot_driver_energy_scatter(sample_files_dict, selected_car)
                            else:
                                sample_ids = random.sample(vehicle_dict[selected_car], len(vehicle_dict[selected_car]))
                                sample_files_dict = {id: [f for f in vehicle_files[selected_car] if id in f] for id in sample_ids}
                                plot_driver_energy_scatter(sample_files_dict, selected_car)
                        elif plot == 4:
                            plot_power_scatter(vehicle_files[selected_car], folder_path)
                        elif plot == 5:
                            plot_energy_dis(vehicle_files[selected_car], selected_car, 'physics')
                        elif plot == 6:
                            plot_energy_dis(vehicle_files[selected_car], selected_car, 'data')
                        elif plot == 7:
                            plot_energy_dis(vehicle_files[selected_car], selected_car, 'hybrid')

                        else:
                            print(f"Invalid choice: {plot}. Please try again.")
                else:
                    continue
                break
        elif task_choice == 6:
            while True:
                print("1: Plotting Residual Contour Graph")
                print("2: Plotting Physics Model Energy Efficiency Graph")
                print("3: Plotting Data Energy Efficiency Graph")
                print("4: Plotting Predicted Energy Efficiency Graph")
                print("5: Driver's Energy Efficiency Graph")
                print("6: Calculating physics-based RMSE & RRMSE")
                print("7: Return to previous menu.")
                print("0: Quitting the program.")
                selections = input("Enter the numbers you want to run, separated by commas (e.g., 1,2,3): ")
                selections_list = selections.split(',')
                for selection in selections_list:
                    try:
                        plot = int(selection.strip())
                    except ValueError:
                        print(f"Invalid input: {selection}. Please enter a valid number.")
                        continue
                    if plot == 7:
                        break
                    elif plot == 0:
                        print("Quitting the program")
                        return
                    for selected_car in selected_cars:
                        if plot == 1:
                            plot_contour2(vehicle_files[selected_car], selected_car)
                        elif plot == 2:
                            plot_2d_histogram(vehicle_files[selected_car], selected_car, 'physics')
                        elif plot == 3:
                            plot_2d_histogram(vehicle_files[selected_car], selected_car)
                        elif plot == 4:
                            plot_2d_histogram(vehicle_files[selected_car], selected_car, 'hybrid')
                        elif plot == 5:
                            required_files = 300
                            all_ids = vehicle_dict[selected_car]

                            while True:
                                sample_ids = random.sample(all_ids, 3)
                                sample_files_dict = {id: [f for f in vehicle_files[selected_car] if id in f] for id in
                                                     sample_ids}
                                total_files = sum(len(files) for files in sample_files_dict.values())
                                if total_files >= required_files:
                                    break
                            plot_2d_histogram(sample_files_dict, selected_car)
                        elif plot == 6:
                            compute_rmse(vehicle_files, selected_car)
                            compute_rrmse(vehicle_files, selected_car)

        elif task_choice == 0:
            print("Quitting the program.")
            return
        else:
            print("Invalid choice. Please try again.")
            continue

if __name__ == "__main__":
    main()
