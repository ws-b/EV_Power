import os
import platform
import time
import random
import numpy as np # GS_Train_Only_XGboost 결과 처리 시 필요할 수 있음

from GS_Merge_Power import process_files_power, select_vehicle
from GS_Functions import get_vehicle_files, compute_mape
from GS_plot import plot_power, plot_energy, plot_energy_scatter, plot_power_scatter, plot_energy_dis, plot_driver_energy_scatter
from GS_vehicle_dict import vehicle_dict

from GS_Train_XGboost import run_workflow as xgb_run_workflow, process_multiple_new_files as xgb_process_multiple_new_files, load_model_and_scaler as xgb_load_model_and_scaler
from GS_Train_Only_XGboost import run_workflow as only_xgb_run_workflow # 함수명 변경 only_run_workflow -> only_xgb_run_workflow
from GS_Train_DNN import run_dnn_workflow
from GS_Train_RF import run_rf_workflow

# GS_Train_Multi__ 모듈 임포트 (이전 GS_Train_Multi 에서 변경)
try:
    from GS_Train_Multi__ import run_evaluate, plot_rmse_results
    GS_TRAIN_MULTI_LOADED = True
except ImportError as e:
    print(f"경고: GS_Train_Multi__ 모듈을 로드할 수 없습니다. 모델 비교 기능(Train Choice 3)이 제한될 수 있습니다. 오류: {e}")
    GS_TRAIN_MULTI_LOADED = False

# GS_Train_LSTM 모듈 임포트 (선택적 로드)
try:
    from GS_Train_LSTM import run_lstm_workflow
    GS_TRAIN_LSTM_LOADED = True
except ImportError:
    print("경고: GS_Train_LSTM.py 모듈을 찾을 수 없습니다. LSTM 관련 기능이 제한될 수 있습니다.")
    GS_TRAIN_LSTM_LOADED = False
    def run_lstm_workflow(*args, **kwargs): # 임시 함수 정의 (호출 시 에러 방지)
        print("오류: LSTM 모듈이 로드되지 않아 run_lstm_workflow를 실행할 수 없습니다.")
        return [], None
    
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
                print("3: Train Models (Comparison with GS_Train_Multi__)")
                print("4: Train DNN Model")
                if GS_TRAIN_LSTM_LOADED: # LSTM 모듈 로드 성공 시에만 메뉴 표시
                    print("5: Train LSTM Model")
                else:
                    print("5: Train LSTM Model (Not available - GS_Train_LSTM.py not found)")
                print("6: Train Random Forest Model")
                print("7: Return to previous menu")
                print("0: Quitting the program")
                try:
                    train_choice = int(input("Enter number you want to run: "))
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

                if train_choice == 7:
                    break
                elif train_choice == 0:
                    print("Quitting the program.")
                    return

                XGB_RESULTS = {}
                ONLY_ML_XGB_RESULTS = {} # 변수명 명확히
                DNN_RESULTS = {}
                LSTM_RESULTS = {}
                RF_RESULTS = {}
                MULTI_MODEL_RESULTS = {} # train_choice == 3 용

                for selected_car in selected_cars:
                    print(f"\n--- Processing car: {selected_car} for Training Option {train_choice} ---")
                    if selected_car not in vehicle_files or not vehicle_files[selected_car]:
                        print(f"No files found for {selected_car}. Skipping.")
                        continue

                    if train_choice == 1:
                        start_time = time.perf_counter()
                        xgb_save_dir = os.path.join(base_model_save_dir, 'XGB_Hybrid')
                        results, scaler = xgb_run_workflow(
                            vehicle_files=vehicle_files,
                            selected_car=selected_car,
                            plot=True, # 개별 모델 학습 시에는 플롯 생성
                            save_dir=xgb_save_dir,
                            predefined_best_params=None
                        )
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"XGB Hybrid Workflow Total Execution Time for {selected_car}: {elapsed_time:.2f} seconds")
                        if results:
                            XGB_RESULTS[selected_car] = results[0] # 첫 번째 결과 항목 사용 (일반적으로 단일 결과)
                            print(f"XGB Hybrid Results for {selected_car}: RMSE={results[0]['rmse']:.4f}, MAPE={results[0]['test_mape']:.2f}%")
                        else:
                            print(f"XGB Hybrid workflow did not produce results for {selected_car}.")

                    elif train_choice == 2:
                        start_time = time.perf_counter()
                        only_xgb_save_dir = os.path.join(base_model_save_dir, 'XGB_Only')
                        try:
                             # only_xgb_run_workflow는 results, scaler 2개만 반환
                             results, scaler = only_xgb_run_workflow(
                                 vehicle_files=vehicle_files,
                                 selected_car=selected_car,
                                 plot=True, # 개별 모델 학습 시에는 플롯 생성
                                 save_dir=only_xgb_save_dir,
                                 predefined_best_params=None
                             )
                             end_time = time.perf_counter()
                             elapsed_time = end_time - start_time
                             print(f"Only XGB Workflow Total Execution Time for {selected_car}: {elapsed_time:.2f} seconds")
                             if results:
                                 # results는 리스트이므로, 첫번째 요소의 값을 사용하거나 평균을 내야 함.
                                 # GS_Train_Only_XGboost.py의 run_workflow는 단일 딕셔너리를 담은 리스트를 반환.
                                 avg_rmse = np.mean([r['rmse'] for r in results if 'rmse' in r])
                                 avg_mape = np.mean([r['test_mape'] for r in results if 'test_mape' in r])
                                 ONLY_ML_XGB_RESULTS[selected_car] = {'rmse': avg_rmse, 'test_mape': avg_mape}
                                 print(f"Only XGB Results for {selected_car}: Avg RMSE={avg_rmse:.4f}, Avg MAPE={avg_mape:.2f}%")
                             else:
                                 print(f"Only XGB workflow did not produce results for {selected_car}.")
                        except NameError:
                             print("Error: 'only_xgb_run_workflow' function is not defined or imported correctly.")
                        except Exception as e:
                             print(f"An error occurred during Only XGB workflow for {selected_car}: {e}")


                    elif train_choice == 3:
                        if not GS_TRAIN_MULTI_LOADED:
                            print("오류: GS_Train_Multi__ 모듈이 로드되지 않아 모델 비교를 실행할 수 없습니다.")
                            continue # 다음 차량으로 또는 메뉴로

                        start_time = time.perf_counter()
                        print(f"[{selected_car}] GS_Train_Multi__ 실행 시작...")
                        try:
                            # run_evaluate는 vehicle_files (모든 차량 데이터)와 selected_car (현재 평가할 차량명)를 받음
                            # 반환값은 해당 차량에 대한 샘플 크기별 모델 평가 결과 딕셔너리
                            results_single_car_multi = run_evaluate(vehicle_files, selected_car)
                            
                            if results_single_car_multi:
                                MULTI_MODEL_RESULTS[selected_car] = results_single_car_multi
                                print(f"[{selected_car}] 모델 비교 평가 완료. 결과 요약:")
                                # 상세 결과는 results_single_car_multi에 저장되어 있으며, JSON 파일로도 저장됨 (GS_Train_Multi__ 내부에서)
                                # 여기서는 간단한 요약만 출력하거나, plot_rmse_results 호출
                                
                                # comparison_figure_save_dir 는 GS_Model_Main.py에서 정의한 경로 사용
                                plot_rmse_results(results_single_car_multi, selected_car, comparison_figure_save_dir)
                                print(f"[{selected_car}] RMSE 비교 그래프가 '{comparison_figure_save_dir}'에 저장되었습니다.")
                            else:
                                print(f"[{selected_car}] 모델 비교 평가에서 결과가 생성되지 않았습니다.")

                        except NameError as ne:
                             print(f"Error: GS_Train_Multi__의 함수(run_evaluate 또는 plot_rmse_results)가 정의되지 않았거나 임포트되지 않았습니다: {ne}")
                        except Exception as e:
                            print(f"An error occurred during Model Comparison for {selected_car}: {e}")

                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"Model Comparison Execution Time for {selected_car}: {elapsed_time:.2f} seconds")

                    elif train_choice == 4:
                        start_time = time.perf_counter()
                        dnn_save_dir = os.path.join(base_model_save_dir, 'DNN')
                        dnn_results, dnn_scaler = run_dnn_workflow(
                            vehicle_files=vehicle_files,
                            selected_car=selected_car,
                            plot=True, # 개별 모델 학습 시에는 플롯 생성
                            save_dir=dnn_save_dir, # 모델 및 스케일러 저장 경로
                            # GS_Train_DNN.py 내부의 그림/결과 저장 경로는 해당 파일 내에서 base_results_save_dir 등을 활용하도록 수정 필요
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

                    elif train_choice == 5:
                        if not GS_TRAIN_LSTM_LOADED:
                            print(f"LSTM 모듈이 로드되지 않아 {selected_car}에 대한 LSTM 모델 학습을 건너뜁니다.")
                            continue
                        start_time = time.perf_counter()
                        lstm_save_dir = os.path.join(base_model_save_dir, 'LSTM')
                        lstm_results, lstm_scaler = run_lstm_workflow(
                            vehicle_files_dict=vehicle_files,       # 수정됨
                            selected_car_name=selected_car,         # 수정됨
                            sequence_length=60,                   # 기본값 사용 또는 필요시 값 전달
                            plot_flag=True,                       # 수정됨 (개별 모델 학습 시에는 플롯 생성)
                            save_models_dir=lstm_save_dir,        # 수정됨 (모델 및 스케일러 저장 경로)
                            existing_best_params=None             # 수정됨 (튜닝 실행 시 None)
                        )
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"LSTM Workflow Total Execution Time for {selected_car}: {elapsed_time:.2f} seconds")
                        if lstm_results:
                            LSTM_RESULTS[selected_car] = lstm_results[0]
                            print(f"LSTM Results for {selected_car}: RMSE={lstm_results[0]['rmse']:.4f}, MAPE={lstm_results[0]['test_mape']:.2f}%")
                        else:
                            print(f"LSTM workflow did not produce results for {selected_car}.")

                    elif train_choice == 6:
                        start_time = time.perf_counter()
                        rf_save_dir = os.path.join(base_model_save_dir, 'RF')
                        rf_results, rf_scaler = run_rf_workflow(
                            vehicle_files=vehicle_files,
                            selected_car=selected_car,
                            plot=True, # 개별 모델 학습 시에는 플롯 생성
                            save_dir=rf_save_dir # 모델 및 스케일러 저장 경로
                            # GS_Train_RF.py 내부의 그림 저장 경로는 해당 파일 내에서 base_results_save_dir 등을 활용하도록 수정 필요
                        )
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"Random Forest Workflow Total Execution Time for {selected_car}: {elapsed_time:.2f} seconds")
                        if rf_results:
                            RF_RESULTS[selected_car] = rf_results[0]
                            print(f"RF Results for {selected_car}: RMSE={rf_results[0]['rmse']:.4f}, MAPE={rf_results[0]['test_mape']:.2f}%")
                        else:
                            print(f"RF workflow did not produce results for {selected_car}.")
                
                # --- 학습 세션 요약 (모든 선택된 차량 처리 후) ---
                if train_choice != 3 : # 개별 모델 학습 시에만 요약 출력
                    print("\n--- Training Session Summary (Individual Models) ---")
                    if XGB_RESULTS: print(f"XGB Hybrid Results: {XGB_RESULTS}")
                    if ONLY_ML_XGB_RESULTS: print(f"Only XGB Results: {ONLY_ML_XGB_RESULTS}")
                    if DNN_RESULTS: print(f"DNN Results: {DNN_RESULTS}")
                    if LSTM_RESULTS: print(f"LSTM Results: {LSTM_RESULTS}") # GS_TRAIN_LSTM_LOADED 확인 후 출력 가능
                    if RF_RESULTS: print(f"Random Forest Results: {RF_RESULTS}")
                elif train_choice == 3 and MULTI_MODEL_RESULTS: # 모델 비교 학습 시 요약
                    print("\n--- Model Comparison Session Summary ---")
                    for car, res_dict in MULTI_MODEL_RESULTS.items():
                        print(f"  Results for {car} stored in JSON and plotted.")
                        # res_dict 내용이 방대하므로, 여기서는 간단히 언급만. 상세 내용은 JSON 파일 및 그래프 참고.
                print("--------------------------------------------------")

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
