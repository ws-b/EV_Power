import os
import platform
import time
import random
from GS_Merge_Power import process_files_power, select_vehicle
from GS_Functions import get_vehicle_files, compute_mape
from GS_plot import plot_power, plot_energy, plot_energy_scatter, plot_power_scatter, plot_energy_dis, plot_driver_energy_scatter
from GS_vehicle_dict import vehicle_dict
from GS_Train_XGboost import run_workflow as xgb_run_workflow, process_multiple_new_files as xgb_process_multiple_new_files, load_model_and_scaler as xgb_load_model_and_scaler
from GS_Train_Only_XGboost import run_workflow as only_run_workflow
from GS_Train_Multi import run_evaluate, plot_rmse_results
from GS_Train_DNN import run_dnn_workflow
from GS_Train_LSTM import run_lstm_workflow
from GS_Train_RF import run_rf_workflow

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

    if platform.system() == "Windows":
        base_processed_path = r'D:\SamsungSTF\Processed_Data'
    elif platform.system() == "Linux":
         base_processed_path = "/home/ubuntu/SamsungSTF/Processed_Data"
    else:
         # 다른 OS에 대한 기본 경로 설정 또는 에러 처리
         base_processed_path = './Processed_Data' # 기본값 또는 에러
         print(f"Warning: Unsupported OS detected ({platform.system()}). Using relative path.")


    folder_path = os.path.join(os.path.normpath(base_processed_path), 'TripByTrip')

    while True:
        print("\n--- Main Menu ---")
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
                # unfiltered_files는 process_files_power에 사용될 수 있음
                unfiltered_files = vehicle_files.get(selected_car, [])
                if not unfiltered_files:
                     print(f"No files found for {selected_car} in task 1.")
                     continue
                process_files_power(unfiltered_files, EV)

        elif task_choice == 2:
            # 모델 저장 기본 경로 설정
            base_model_save_dir = os.path.join(os.path.dirname(folder_path), 'Models')

            while True:
                print("\n--- Train Model Menu ---")
                print("1: Hybrid(XGB) Model")
                print("2: Only ML(XGB) Model")
                print("3: Train Models (Comparison)") # Requires GS_Train_Multi implementation
                print("4: Train DNN Model")
                print("5: Train LSTM Model")          # LSTM 옵션 추가
                print("6: Train Random Forest Model") # RF 옵션 번호 변경
                print("7: Return to previous menu")   # 이전 메뉴 옵션 번호 변경
                print("0: Quitting the program")
                try:
                    train_choice = int(input("Enter number you want to run: "))
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

                # 메뉴 번호에 따른 분기 업데이트
                if train_choice == 7: # 이전 메뉴로 돌아가기
                    break
                elif train_choice == 0: # 프로그램 종료
                    print("Quitting the program.")
                    return

                # 결과 저장을 위한 딕셔너리 초기화
                XGB_RESULTS = {}
                ONLY_ML_RESULTS = {}
                DNN_RESULTS = {}
                LSTM_RESULTS = {}
                RF_RESULTS = {}

                for selected_car in selected_cars:
                    print(f"\n--- Processing car: {selected_car} for Training Option {train_choice} ---")
                    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
                        print(f"No files found for {selected_car}. Skipping.")
                        continue

                    # 각 모델 선택 시 처리
                    if train_choice == 1:
                        start_time = time.perf_counter()
                        xgb_save_dir = os.path.join(base_model_save_dir, 'XGB_Hybrid')
                        results, scaler = xgb_run_workflow(
                            vehicle_files=vehicle_files,
                            selected_car=selected_car,
                            plot=True,
                            save_dir=xgb_save_dir,
                            predefined_best_params=None
                        )
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"XGB Hybrid Workflow Total Execution Time for {selected_car}: {elapsed_time:.2f} seconds")
                        if results:
                            XGB_RESULTS[selected_car] = results[0]
                            print(f"XGB Hybrid Results for {selected_car}: RMSE={results[0]['rmse']:.4f}, MAPE={results[0]['test_mape']:.2f}%")
                        else:
                            print(f"XGB Hybrid workflow did not produce results for {selected_car}.")

                    elif train_choice == 2:
                        start_time = time.perf_counter()
                        only_xgb_save_dir = os.path.join(base_model_save_dir, 'XGB_Only')
                        try:
                             results, scaler, _ = only_run_workflow(
                                 vehicle_files=vehicle_files,
                                 selected_car=selected_car,
                                 plot=True,
                                 save_dir=only_xgb_save_dir,
                                 predefined_best_params=None
                             )
                             end_time = time.perf_counter()
                             elapsed_time = end_time - start_time
                             print(f"Only XGB Workflow Total Execution Time for {selected_car}: {elapsed_time:.2f} seconds")
                             if results:
                                 avg_rmse = np.mean([r['rmse'] for r in results])
                                 avg_mape = np.mean([r['test_mape'] for r in results])
                                 ONLY_ML_RESULTS[selected_car] = {'rmse': avg_rmse, 'test_mape': avg_mape}
                                 print(f"Only XGB Results for {selected_car}: Avg RMSE={avg_rmse:.4f}, Avg MAPE={avg_mape:.2f}%")
                             else:
                                 print(f"Only XGB workflow did not produce results for {selected_car}.")
                        except NameError:
                             print("Error: 'only_run_workflow' function is not defined or imported.")
                        except Exception as e:
                             print(f"An error occurred during Only XGB workflow: {e}")


                    elif train_choice == 3:
                        start_time = time.perf_counter()
                        comparison_save_path = os.path.join(os.path.dirname(folder_path), 'Figures', 'Result') # 결과 그림 저장 경로
                        try:
                            results_dict = run_evaluate(vehicle_files, selected_car)
                            if not os.path.exists(comparison_save_path):
                                 os.makedirs(comparison_save_path)
                            plot_rmse_results(results_dict, selected_car, comparison_save_path)
                        except NameError as ne:
                             print(f"Error: Function not defined or imported for comparison: {ne}")
                        except Exception as e:
                            print(f"An error occurred during Model Comparison: {e}")

                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"Model Comparison Execution Time for {selected_car}: {elapsed_time:.2f} seconds")

                    elif train_choice == 4: # DNN 모델 학습습
                        start_time = time.perf_counter()
                        dnn_save_dir = os.path.join(base_model_save_dir, 'DNN')
                        dnn_results, dnn_scaler = run_dnn_workflow(
                            vehicle_files=vehicle_files,
                            selected_car=selected_car,
                            plot=True,
                            save_dir=dnn_save_dir,
                            predefined_best_params=None
                        )
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"DNN Workflow Total Execution Time for {selected_car}: {elapsed_time:.2f} seconds")
                        if dnn_results:
                            DNN_RESULTS[selected_car] = dnn_results[0]
                            print(f"DNN Results for {selected_car}: RMSE={dnn_results[0]['rmse']:.4f}, MAPE={dnn_results[0]['test_mape']:.2f}%")
                        else:
                            print(f"DNN workflow did not produce results for {selected_car}.")

                    elif train_choice == 5: # LSTM 모델 학습
                        start_time = time.perf_counter()
                        lstm_save_dir = os.path.join(base_model_save_dir, 'LSTM')
                        lstm_results, lstm_scaler = run_lstm_workflow(
                            vehicle_files=vehicle_files,
                            selected_car=selected_car,
                            sequence_length=60, # 시퀀스 길이 (필요시 조정)
                            plot=True,
                            save_dir=lstm_save_dir,
                            predefined_best_params=None
                        )
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"LSTM Workflow Total Execution Time for {selected_car}: {elapsed_time:.2f} seconds")
                        if lstm_results:
                            LSTM_RESULTS[selected_car] = lstm_results[0]
                            print(f"LSTM Results for {selected_car}: RMSE={lstm_results[0]['rmse']:.4f}, MAPE={lstm_results[0]['test_mape']:.2f}%")
                        else:
                            print(f"LSTM workflow did not produce results for {selected_car}.")

                    elif train_choice == 6: # Random Forest 모델 학습
                        start_time = time.perf_counter()
                        rf_save_dir = os.path.join(base_model_save_dir, 'RF')
                        rf_results, rf_scaler = run_rf_workflow(
                            vehicle_files=vehicle_files,
                            selected_car=selected_car,
                            plot=True,
                            save_dir=rf_save_dir
                        )
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"Random Forest Workflow Total Execution Time for {selected_car}: {elapsed_time:.2f} seconds")
                        if rf_results:
                            RF_RESULTS[selected_car] = rf_results[0]
                            print(f"RF Results for {selected_car}: RMSE={rf_results[0]['rmse']:.4f}, MAPE={rf_results[0]['test_mape']:.2f}%")
                        else:
                            print(f"RF workflow did not produce results for {selected_car}.")

                # 선택한 모든 차량에 대한 학습 완료 후 결과 요약 출력
                print("\n--- Training Session Summary ---")
                if XGB_RESULTS: print(f"XGB Hybrid Results: {XGB_RESULTS}")
                if ONLY_ML_RESULTS: print(f"Only XGB Results: {ONLY_ML_RESULTS}")
                if DNN_RESULTS: print(f"DNN Results: {DNN_RESULTS}")
                if LSTM_RESULTS: print(f"LSTM Results: {LSTM_RESULTS}")
                if RF_RESULTS: print(f"Random Forest Results: {RF_RESULTS}")
                print("---------------------------------")

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
