import os
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from GS_preprocessing import load_data_by_vehicle
from GS_Merge_Power import process_files_power, select_vehicle, compute_rrmse
from GS_plot import plot_power, plot_energy, plot_energy_scatter, plot_power_scatter, plot_energy_dis, plot_driver_energy_scatter, plot_contour2, plot_2d_histogram
from GS_vehicle_dict import vehicle_dict
from GS_Train_XGboost import cross_validate as xgb_cross_validate, add_predicted_power_column as xgb_add_predicted_power_column
from GS_Train_Only_XGboost import cross_validate as only_xgb_validate
from GS_Train_LinearR import cross_validate as lr_cross_validate, add_predicted_power_column as lr_add_predicted_power_column

def get_vehicle_files(car_options, folder_path, vehicle_dict):
    selected_cars = []
    vehicle_files = {}
    while True:
        print("Available Cars:")
        for idx, car_name in car_options.items():
            print(f"{idx}: {car_name}")
        print("0: Done selecting cars")
        car_input = input("Select Cars you want to include 콤마로 구분 (e.g.: 1,2,3): ")

        try:
            car_list = [int(car.strip()) for car in car_input.split(',')]
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
            continue

        if 0 in car_list:
            car_list.remove(0)
            for car in car_list:
                if car in car_options:
                    selected_car = car_options[car]
                    if selected_car not in selected_cars:
                        selected_cars.append(selected_car)
                        vehicle_files = vehicle_files | load_data_by_vehicle(folder_path, vehicle_dict, selected_car)
                else:
                    print(f"Invalid choice: {car}. Please try again.")
            break
        else:
            for car in car_list:
                if car in car_options:
                    selected_car = car_options[car]
                    if selected_car not in selected_cars:
                        selected_cars.append(selected_car)
                        vehicle_files= vehicle_files | load_data_by_vehicle(folder_path, vehicle_dict, selected_car)
                else:
                    print(f"Invalid choice: {car}. Please try again.")

    return selected_cars, vehicle_files

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

                process_files_power(vehicle_files.get(selected_car, []), EV)

        elif task_choice == 2:
            save_dir = os.path.join(os.path.dirname(folder_path), 'Models')
            while True:
                print("1: Physics-based Equation calculate relative RMSE")
                print("2: Train Model using XGBoost")
                print("3: Train Model using Linear Regression")
                print("4: Train Model using Only ML")
                print("5: Train Models with varying vehicle_files sizes")
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
                PE_RRMSE = {}
                XGB_RRMSE = {}
                LR_RRMSE = {}
                ONLY_RRMSE = {}
                results_dict = {}

                Physics_Only_RRMSE = {
                    'EV6': 1.85,
                    'Ioniq5': 1.59,
                    'KonaEV': 1.67,
                    'NiroEV': 2.04,
                    'GV60': 2.16,
                    'Ioniq6': 1.70
                }

                for selected_car in selected_cars:
                    if train_choice == 1:
                        rrmse = compute_rrmse(vehicle_files, selected_car)
                        PE_RRMSE[selected_car] = rrmse
                    if train_choice == 2:
                        results, scaler = xgb_cross_validate(vehicle_files, selected_car, save_dir=save_dir)

                        if results:
                            rrmse_values = []
                            for fold_num, rrmse in results:
                                print(f"Fold: {fold_num}, RRMSE: {rrmse}")
                                rrmse_values.append(rrmse)
                            XGB_RRMSE[selected_car] = np.median(rrmse_values)
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 3:
                        results, scaler = lr_cross_validate(vehicle_files, selected_car, save_dir=save_dir)

                        if results:
                            rrmse_values = []
                            for fold_num, rrmse in results:
                                print(f"Fold: {fold_num}, RRMSE: {rrmse}")
                                rrmse_values.append(rrmse)
                            # Store the minimum RMSE in dictionaries
                            LR_RRMSE[selected_car] = np.median(rrmse_values)
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 4:
                        results, scaler = only_xgb_validate(vehicle_files, selected_car, save_dir=save_dir)

                        if results:
                            rrmse_values = []
                            for fold_num, rrmse in results:
                                print(f"Fold: {fold_num}, RRMSE: {rrmse}")
                                rrmse_values.append(rrmse)
                            ONLY_RRMSE[selected_car] = np.median(rrmse_values)
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")

                    if train_choice == 5:
                        vehicle_file_sizes = [5, 7, 10, 20, 50, 100, 200, 500,
                                              1000, 2000, 3000, 5000, 10000]

                        results_dict[selected_car] = {}
                        max_samples = len(vehicle_files[selected_car])

                        filtered_vehicle_file_sizes = [size for size in vehicle_file_sizes if size <= max_samples]

                        for size in filtered_vehicle_file_sizes:
                            if size < 50:
                                iterations = 10
                            elif 50 <= size <= 100:
                                iterations = 6
                            else:
                                iterations = 1

                            for iteration in range(iterations):
                                sampled_files = random.sample(vehicle_files[selected_car], size)
                                sampled_vehicle_files = {selected_car: sampled_files}

                                results, scaler = xgb_cross_validate(sampled_vehicle_files, selected_car, save_dir=None)
                                if results:
                                    rrmse_values = [rrmse for fold_num, rrmse in results]
                                if size not in results_dict[selected_car]:
                                    results_dict[selected_car][size] = []
                                results_dict[selected_car][size].append({
                                    'model': 'Hybrid Model(XGBoost)',
                                    'selected_car': selected_car,
                                    'rrmse': rrmse_values
                                })

                                results, scaler = lr_cross_validate(sampled_vehicle_files, selected_car, save_dir=None)
                                if results:
                                    rrmse_values = [rrmse for fold_num, rrmse in results]
                                    if size not in results_dict[selected_car]:
                                        results_dict[selected_car][size] = []
                                    results_dict[selected_car][size].append({
                                        'model': 'Hybrid Model(Linear Regression)',
                                        'selected_car': selected_car,
                                        'rrmse': rrmse_values
                                    })

                                results, scaler = only_xgb_validate(sampled_vehicle_files, selected_car, save_dir=None)
                                if results:
                                    rrmse_values = [rrmse for fold_num, rrmse in results]
                                    if size not in results_dict[selected_car]:
                                        results_dict[selected_car][size] = []
                                    results_dict[selected_car][size].append({
                                        'model': 'Only ML(XGBoost)',
                                        'selected_car': selected_car,
                                        'rrmse': rrmse_values
                                    })

                print(results_dict)

                for selected_car in selected_cars:
                    results_dict[selected_car] = results
                    sizes = sorted(results.keys())
                    Physics_Only_RRMSE[selected_car] = physics_only_rrmse

                    xgb_rrmse = []
                    lr_rrmse = []
                    only_ml_rrmse = []

                    for size in sizes:
                        xgb_values = [item for result in results[size] if result['model'] == 'Hybrid Model(XGBoost)' for
                                      item in result['rrmse']]
                        lr_values = [item for result in results[size] if
                                     result['model'] == 'Hybrid Model(Linear Regression)' for item in result['rrmse']]
                        only_ml_values = [item for result in results[size] if result['model'] == 'Only ML(XGBoost)' for
                                          item in result['rrmse']]

                        if xgb_values:
                            xgb_rrmse.append(np.mean(xgb_values))
                        if lr_values:
                            lr_rrmse.append(np.mean(lr_values))
                        if only_ml_values:
                            only_ml_rrmse.append(np.mean(only_ml_values))

                    plt.figure(figsize=(10, 6))
                    plt.plot(sizes, xgb_rrmse, label='Hybrid Model(XGBoost)', marker='o')
                    plt.plot(sizes, lr_rrmse, label='Hybrid Model(Linear Regression)', marker='o')
                    plt.plot(sizes, only_ml_rrmse, label='Only ML(XGBoost)', marker='o')
                    plt.axhline(y=physics_only_rrmse, color='r', linestyle='--', label='Physics Model')

                    plt.xlabel('Number of Trips')
                    plt.ylabel('RRMSE')
                    plt.title(f'RRMSE vs Number of Trips for {selected_car}')
                    plt.legend()
                    plt.grid(True)

                    plt.xscale('symlog', linthresh=20)
                    plt.xticks(sizes, [str(size) for size in sizes], rotation=45)
                    plt.xlim(min(sizes) - 1, max(sizes) + 1000)
                    plt.show()

                print(f"XGB RRMSE: {XGB_RRMSE}")
                print(f"LR RRMSE: {LR_RRMSE}")
                print(f"ONLY ML RRMSE: {ONLY_RRMSE}")

        elif task_choice == 3:
            while True:
                print("1: XGBoost Model")
                print("2: Linear Regression")
                print("3: Return to previous menu")
                print("0: Quitting the program")
                try:
                    pred_choice = int(input("Enter number you want to run: "))
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

                if pred_choice == 3:
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
                        model_path = os.path.join(os.path.dirname(folder_path), 'Models', f'SVR_best_model_{selected_car}.json')
                        scaler_path = os.path.join(os.path.dirname(folder_path), 'Models', f'SVR_scaler_{selected_car}.pkl')

                        if not vehicle_files[selected_car]:
                            print(f"No files to process for the selected vehicle: {selected_car}")
                            continue

                        # Load the scaler
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)

                        svr_add_predicted_power_column(vehicle_files[selected_car], model_path, scaler)


        elif task_choice == 4:
            while True:
                print("1: Plotting Stacked Power Plot Term by Term")
                print("2: Plotting Model's Power Graph")
                print("3: Plotting Data's Power Graph")
                print("4: Plotting Power Comparison Graph")
                print("5: ")
                print("6: Plotting Delta Altitude and Difference Graph")
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
                            plot_power(sample_files, selected_car, 'model')
                        elif plot == 3:
                            plot_power(sample_files, selected_car, 'data')
                        elif plot == 4:
                            plot_power(sample_files, selected_car, 'comparison')
                        elif plot == 5:
                            break
                        elif plot == 6:
                            plot_power(sample_alt_files, selected_car, 'd_altitude')
                        elif plot == 7:
                            plot_energy(sample_files, selected_car, 'model')
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
                print("5: Plotting Model Energy Distribution Graph")
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
                            plot_energy_scatter(vehicle_files[selected_car], selected_car, 'model')
                        elif plot == 2:
                            plot_energy_scatter(vehicle_files[selected_car], selected_car, 'learning')
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
                            plot_energy_dis(vehicle_files[selected_car], selected_car, 'model')
                        elif plot == 6:
                            plot_energy_dis(vehicle_files[selected_car], selected_car, 'data')
                        elif plot == 7:
                            plot_energy_dis(vehicle_files[selected_car], selected_car, 'learning')

                        else:
                            print(f"Invalid choice: {plot}. Please try again.")
                else:
                    continue
                break
        elif task_choice == 6:
            while True:
                print("1: Plotting Residual Contour Graph")
                print("2: Plotting Model Energy Efficiency Graph")
                print("3: Plotting Data Energy Efficiency Graph")
                print("4: Plotting Predicted Energy Efficiency Graph")
                print("5: Driver's Energy Efficiency Graph")
                print("6: ")
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
                            plot_2d_histogram(vehicle_files[selected_car], selected_car, 'model')
                        elif plot == 3:
                            plot_2d_histogram(vehicle_files[selected_car], selected_car)
                        elif plot == 4:
                            plot_2d_histogram(vehicle_files[selected_car], selected_car, 'learning')
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
                            break
        elif task_choice == 0:
            print("Quitting the program.")
            return
        else:
            print("Invalid choice. Please try again.")
            continue

if __name__ == "__main__":
    main()
