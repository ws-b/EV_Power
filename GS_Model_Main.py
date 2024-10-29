import os
import time
import random
from GS_Merge_Power import process_files_power, select_vehicle
from GS_Functions import get_vehicle_files, compute_mape
from GS_plot import plot_power, plot_energy, plot_energy_scatter, plot_power_scatter, plot_energy_dis, plot_driver_energy_scatter
from GS_vehicle_dict import vehicle_dict
from GS_Train_XGboost import cross_validate as xgb_cross_validate, process_multiple_new_files as xgb_process_multiple_new_files, load_model_and_scaler as xgb_load_model_and_scaler
from GS_Train_Only_XGboost import cross_validate as only_xgb_validate
from GS_Train_Only_LR import cross_validate as only_lr_validate
from GS_Train_LinearR import cross_validate as lr_cross_validate
from GS_Train_LightGBM import cross_validate as lgbm_cross_validate
from GS_Train_Multi import run_evaluate, plot_rmse_results
from GS_Train_Multi_Sample import run_evaluate_20_50
from GS_Train_SVR import cross_validate as svr_cross_validate
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
                print("5: Train Models")
                print("6: Hybrid(SVR) Model")
                print("7: Only ML(LR) Model")
                print("8: 20-50 Sample Train Models")
                print("9: Return to previous menu")
                print("0: Quitting the program")
                try:
                    train_choice = int(input("Enter number you want to run: "))
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

                if train_choice == 9:
                    break
                elif train_choice == 0:
                    print("Quitting the program.")
                    return
                XGB = {}
                LR = {}
                LGBM = {}
                ONLY_ML = {}
                ONLY_ML_LR = {}
                SVR = {}

                for selected_car in selected_cars:
                    XGB[selected_car] = {}
                    if train_choice == 1:
                        start_time = time.perf_counter()

                        results, scaler, _ = xgb_cross_validate(vehicle_files, selected_car, None, True,  save_dir=None)

                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"Total Execution Time: {elapsed_time:.2f} seconds")

                        if results:
                            rmse_values = []
                            mape_values = []
                            rrmse_values = []

                            for res in results:
                                fold_num = res['fold']
                                rmse = res['rmse']
                                rrmse_test = res['test_rrmse']
                                mape_test = res['test_mape']

                                print(f"Fold: {fold_num}, RMSE: {rmse}, RRMSE: {rrmse_test}, MAPE: {mape_test}")

                                rmse_values.append(rmse)
                                mape_values.append(mape_test)
                                rrmse_values.append(rrmse_test)

                            # XGB 딕셔너리에 결과 저장
                            XGB[selected_car] = {
                                'RMSE': rmse_values,
                                'RRMSE': rrmse_values,
                                'MAPE': mape_values
                            }

                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 2:
                        start_time = time.perf_counter()

                        results, scaler = lr_cross_validate(vehicle_files, selected_car)

                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"Total Execution Time: {elapsed_time:.2f} seconds")

                        if results:
                            rmse_values = []
                            mape_values = []
                            rrmse_values = []
                            for res in results:
                                fold_num = res['fold']
                                rmse = res['rmse']
                                rrmse_test = res['test_rrmse']
                                mape_test = res['test_mape']

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
                        start_time = time.perf_counter()

                        results, scaler, _ = lgbm_cross_validate(vehicle_files, selected_car, None, None)

                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"Total Execution Time: {elapsed_time:.2f} seconds")

                        if results:
                            rmse_values = []
                            mape_values = []
                            rrmse_values = []
                            for res in results:
                                fold_num = res['fold']
                                rmse = res['rmse']
                                rrmse_test = res['test_rrmse']
                                mape_test = res['test_mape']

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
                        start_time = time.perf_counter()

                        results, scaler, _ = only_xgb_validate(vehicle_files, selected_car, None, None)

                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"Total Execution Time: {elapsed_time:.2f} seconds")

                        if results:
                            mape_values = []
                            rmse_values = []
                            rrmse_values = []
                            for res in results:
                                fold_num = res['fold']
                                rmse = res['rmse']
                                rrmse_test = res['test_rrmse']
                                mape_test = res['test_mape']

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
                        start_time = time.perf_counter()

                        save_path = r"C:\Users\BSL\Desktop\Figures\Result"

                        results_dict = run_evaluate(vehicle_files, selected_car)
                        plot_rmse_results(results_dict, selected_car, save_path)

                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time

                        print(f"Total Execution Time: {elapsed_time:.2f} seconds")

                    if train_choice == 6:
                        start_time = time.perf_counter()

                        predefined_params = {
                            'C': 988.20455,
                            'epsilon': 0.02677,
                            'gamma': 'auto',
                            'kernel': 'rbf'
                        }
                        results, scaler, _ = svr_cross_validate(vehicle_files, selected_car, predefined_params, False)

                        # 종료 시간 기록
                        end_time = time.perf_counter()
                        # 총 실행 시간 계산
                        elapsed_time = end_time - start_time

                        # 실행 시간 출력
                        print(f"Total Execution Time: {elapsed_time:.2f} seconds")

                        if results:
                            rmse_values = []
                            mape_values = []
                            rrmse_values = []
                            for res in results:
                                fold_num = res['fold']
                                rmse = res['rmse']
                                rrmse_test = res['test_rrmse']
                                mape_test = res['test_mape']

                                print(f"Fold: {fold_num}, RMSE: {rmse}, RRMSE: {rrmse_test}, MAPE: {mape_test}")

                                rmse_values.append(rmse)
                                mape_values.append(mape_test)
                                rrmse_values.append(rrmse_test)

                            SVR[selected_car] = {
                                'RMSE': rmse_values,
                                'RRMSE': rrmse_values,
                                'MAPE': mape_values
                            }

                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 7:
                        start_time = time.perf_counter()

                        results, scaler = only_lr_validate(vehicle_files, selected_car)

                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"Total Execution Time: {elapsed_time:.2f} seconds")

                        if results:
                            mape_values = []
                            rmse_values = []
                            rrmse_values = []
                            for res in results:
                                fold_num = res['fold']
                                rmse = res['rmse']
                                rrmse_test = res['test_rrmse']
                                mape_test = res['test_mape']

                                print(f"Fold: {fold_num}, RMSE: {rmse}, RRMSE: {rrmse_test}, MAPE: {mape_test}")

                                rmse_values.append(rmse)
                                mape_values.append(mape_test)
                                rrmse_values.append(rrmse_test)

                            ONLY_ML_LR[selected_car] = {
                                'RMSE': rmse_values,
                                'RRMSE': rrmse_values,
                                'MAPE': mape_values
                            }
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 8:
                        start_time = time.perf_counter()

                        results_dict = run_evaluate_20_50(vehicle_files, selected_car)

                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time

                        print(f"Total Execution Time: {elapsed_time:.2f} seconds")
                print(f"XGB RRMSE & MAPE: {XGB}")
                print(f"LR RRMSE & MAPE: {LR}")
                print(f"LGBM RRMSE & MAPE: {LGBM}")
                print(f"ONLY ML RRMSE & MAPE: {ONLY_ML}")
                print(f"Only ML LR RRMSE & MAPE: {ONLY_ML_LR}")
                print(f"SVR RRMSE & MAPE: {SVR}")

        elif task_choice == 3:
            for selected_car in selected_cars:
                model_path = os.path.join(os.path.dirname(folder_path), 'Models',
                                          f'XGB_best_model_{selected_car}.model')
                scaler_path = os.path.join(os.path.dirname(folder_path), 'Models', f'XGB_scaler_{selected_car}.pkl')

                if not vehicle_files[selected_car]:
                    print(f"No files to process for the selected vehicle: {selected_car}")
                    continue

                model, scaler = xgb_load_model_and_scaler(model_path, scaler_path)

                xgb_process_multiple_new_files(
                    vehicle_files[selected_car],
                    model=model,
                    scaler=scaler
                )

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
                        filtered_files = [f for f in vehicle_files.get(selected_car, []) if
                                          f.endswith('.csv') and 'bms' in f and 'altitude' in f]
                        if plot == 1:
                            plot_power(filtered_files[:5], selected_car, 'stacked')
                        elif plot == 2:
                            plot_power(sample_files, selected_car, 'physics')
                        elif plot == 3:
                            plot_power(sample_files, selected_car, 'data')
                        elif plot == 4:
                            plot_power(filtered_files[:5], selected_car, 'comparison')
                        elif plot == 5:
                            plot_power(sample_files, selected_car, 'hybrid')
                        elif plot == 6:
                            plot_power(sample_alt_files, selected_car, 'altitude')
                        elif plot == 7:
                            plot_energy(sample_files, selected_car, 'physics')
                        elif plot == 8:
                            plot_energy(sample_files, selected_car, 'data')
                        elif plot == 9:
                            plot_energy(filtered_files[:5], selected_car, 'comparison')
                        elif plot == 10:
                            plot_energy(filtered_files[:5], selected_car, 'altitude')
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
                print("2: Driver's Energy Efficiency Graph")
                print("3: Calculating MAPE, RRMSE for Physics Model")
                print("4: Return to previous menu.")
                print("0: Quitting the program.")
                selections = input("Enter the numbers you want to run, separated by commas (e.g., 1,2,3): ")
                selections_list = selections.split(',')
                for selection in selections_list:
                    try:
                        plot = int(selection.strip())
                    except ValueError:
                        print(f"Invalid input: {selection}. Please enter a valid number.")
                        continue
                    if plot == 4:
                        break
                    elif plot == 0:
                        print("Quitting the program")
                        return
                    for selected_car in selected_cars:
                        if plot == 1:
                            plot_contour2(vehicle_files[selected_car], selected_car)
                        elif plot == 2:
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
                        elif plot == 3:
                            compute_mape(vehicle_files, selected_car)

        elif task_choice == 0:
            print("Quitting the program.")
            return
        else:
            print("Invalid choice. Please try again.")
            continue

if __name__ == "__main__":
    main()
