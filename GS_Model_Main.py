import os
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from GS_Merge_Power import process_files_power, select_vehicle
from GS_Functions import get_vehicle_files, compute_mape_rrmse
from GS_plot import plot_power, plot_energy, plot_energy_scatter, plot_power_scatter, plot_energy_dis, plot_driver_energy_scatter, plot_2d_histogram
from GS_vehicle_dict import vehicle_dict
from GS_Train_XGboost import cross_validate as xgb_cross_validate, add_predicted_power_column as xgb_add_predicted_power_column
from GS_Train_XGboost_Kmeans import cross_validate as xgb_kmeans_cross_validate
from GS_Train_Only_XGboost import cross_validate as only_xgb_validate
from GS_Train_LinearR import cross_validate as lr_cross_validate
from GS_Train_LightGBM import cross_validate as lgbm_cross_validate
from GS_Train_Multi import run_evaluate, plot_mape_results, plot_rrmse_results
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
                print("3: Hybrid(LGBM) Model")
                print("4: Only ML(XGB) Model")
                print("5: Hybrid(XGB, K-means) Model")
                print("7: Train Models")
                print("8: Return to previous menu")
                print("0: Quitting the program")
                try:
                    train_choice = int(input("Enter number you want to run: "))
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

                if train_choice == 8:
                    break
                elif train_choice == 0:
                    print("Quitting the program.")
                    return
                XGB = {}
                LR = {}
                LGBM = {}
                ONLY_ML = {}
                
                for selected_car in selected_cars:
                    XGB[selected_car] = {}
                    if train_choice == 1:
                        results, scaler, _ = xgb_cross_validate(vehicle_files, selected_car, None, True,  save_dir=save_dir)

                        if results:
                            rmse_values = []
                            mape_values = []
                            rrmse_values = []
                            for fold_num, rmse, _, _, rrmse_test, mape_test in results:
                                print(f"Fold: {fold_num}, RMSE: {rmse}, RRMSE: {rrmse_test}, MAPE: {mape_test}")
                                rmse_values.append(rmse)
                                mape_values.append(mape_test)
                                rrmse_values.append(rrmse_test)
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
                            rmse_values = []
                            mape_values = []
                            rrmse_values = []
                            for fold_num, rmse, _, _, rrmse_test, mape_test in results:
                                print(f"Fold: {fold_num}, RMSE: {rmse}, RRMSE: {rrmse_test}, MAPE: {mape_test}")
                                rmse_values.append(rmse)
                                mape_values.append(mape_test)
                                rrmse_values.append(rrmse_test)
                            LR[selected_car] = {
                                'RMSE': rmse_values,
                                'RRMSE': rrmse_values,
                                'MAPE': mape_values
                            }
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 3:
                        results, scaler, _ = lgbm_cross_validate(vehicle_files, selected_car, None, True,
                                                                 save_dir=save_dir)
                        if results:
                            rmse_values = []
                            mape_values = []
                            rrmse_values = []
                            for fold_num, rmse, _, _, rrmse_test, mape_test in results:
                                print(f"Fold: {fold_num}, RMSE: {rmse}, RRMSE: {rrmse_test}, MAPE: {mape_test}")
                                rmse_values.append(rmse)
                                mape_values.append(mape_test)
                                rrmse_values.append(rrmse_test)
                            LGBM[selected_car] = {
                                'RMSE': rmse_values,
                                'RRMSE': rrmse_values,
                                'MAPE': mape_values
                            }
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")

                    if train_choice == 4:
                        results, scaler, _ = only_xgb_validate(vehicle_files, selected_car, None, True, save_dir=save_dir)

                        if results:
                            mape_values = []
                            rmse_values = []
                            rrmse_values = []
                            for fold_num, rmse, _, _, rrmse_test, mape_test in results:
                                print(f"Fold: {fold_num}, RMSE: {rmse}, RRMSE: {rrmse_test}, MAPE: {mape_test}")
                                rmse_values.append(rmse)
                                mape_values.append(mape_test)
                                rrmse_values.append(rrmse_test)
                            ONLY_ML[selected_car] = {
                                'RMSE': rmse_values,
                                'RRMSE': rrmse_values,
                                'MAPE': mape_values
                            }
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 5:
                        for n_clusters in [3,4,5,6,7,8]:
                            results, scaler, residual_scaler, kmeans = xgb_kmeans_cross_validate(vehicle_files, selected_car, None, True,  save_dir=save_dir, n_clusters = n_clusters)

                            if results:
                                rmse_values = []
                                mape_values = []
                                rrmse_values = []
                                for fold_num, rmse, _, _, rrmse_test, mape_test in results:
                                    print(f"Fold: {fold_num}, RMSE: {rmse}, RRMSE: {rrmse_test}, MAPE: {mape_test}")
                                    rmse_values.append(rmse)
                                    mape_values.append(mape_test)
                                    rrmse_values.append(rrmse_test)
                                XGB[selected_car][n_clusters] = {
                                    'RMSE': rmse_values,
                                    'RRMSE': rrmse_values,
                                    'MAPE': mape_values
                                }

                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 7:
                        save_path = r"C:\Users\BSL\Desktop\Figures\Result"
                        results_dict = run_evaluate(vehicle_files, selected_car)
                        plot_mape_results(results_dict, selected_car, save_path)
                        plot_rrmse_results(results_dict, selected_car, save_path)

                print(f"XGB RRMSE & MAPE: {XGB}")
                print(f"LR RRMSE & MAPE: {LR}")
                print(f"LGBM RRMSE & MAPE: {LGBM}")
                print(f"ONLY ML RRMSE & MAPE: {ONLY_ML}")

        elif task_choice == 3:
            while True:
                print("1: XGBoost Model")
                print("2: Return to previous menu")
                print("0: Quitting the program")
                try:
                    pred_choice = int(input("Enter number you want to run: "))
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

                if pred_choice == 2:
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
                print("1: Plotting Physics-based Model Energy Scatter Graph")
                print("2: Plotting Hybrid Model Scatter Graph")
                print("3: Plotting Individual Driver's Scatter Graph")
                print("4: Plotting Power and Delta_altitude Graph")
                print("5: Plotting Physics-based Model Energy Distribution Graph")
                print("6: Plotting Data Energy Distribution Graph")
                print("7: Plotting Hybrid Model Energy Distribution Graph")
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
                print("6: Calculating MAPE, RRMSE for Physics Model")
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
                            compute_mape_rrmse(vehicle_files, selected_car)

        elif task_choice == 0:
            print("Quitting the program.")
            return
        else:
            print("Invalid choice. Please try again.")
            continue

if __name__ == "__main__":
    main()
