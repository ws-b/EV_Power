import os
import random
import pickle
from GS_preprocessing import load_data_by_vehicle
from GS_Merge_Power import process_files_power, select_vehicle
from GS_plot import plot_power, plot_energy, plot_energy_scatter, plot_power_scatter, plot_energy_dis, plot_driver_energy_scatter, plot_contour2, plot_2d_histogram
from GS_vehicle_dict import vehicle_dict
from GS_Train_XGboost import cross_validate as xgb_cross_validate, add_predicted_power_column as xgb_add_predicted_power_column
from GS_Train_LinearR import cross_validate as lr_cross_validate, add_predicted_power_column as lr_add_predicted_power_column
from GS_Train_SVR import cross_validate as svr_cross_validate, add_predicted_power_column as svr_add_predicted_power_column

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
        2: 'Ionic5',
        3: 'KonaEV',
        4: 'NiroEV',
        5: 'GV60',
        6: 'Ionic6',
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
                process_files_power(vehicle_files.get(selected_car, []), EV)

        elif task_choice == 2:
            save_dir = os.path.join(os.path.dirname(folder_path), 'Models')
            while True:
                print("1: Train Model using XGBoost")
                print("2: Train Model using Linear Regression")
                print("3: Train Model using SVM")
                print("4: Return to previous menu")
                print("0: Quitting the program")
                try:
                    train_choice = int(input("Enter number you want to run: "))
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

                if train_choice == 4:
                    break
                elif train_choice == 0:
                    print("Quitting the program.")
                    return
                XGB_RMSE = {}
                LR_RMSE = {}
                SVR_RMSE = {}
                for selected_car in selected_cars:
                    if train_choice == 1:
                        results, scaler = xgb_cross_validate(vehicle_files, selected_car, save_dir=save_dir)

                        # Print overall results
                        if results:
                            min_rmse = float('inf')
                            for fold_num, rmse, mae in results:
                                print(f"Fold: {fold_num}, RMSE: {rmse}, MAE: {mae}")
                                if rmse < min_rmse:
                                    min_rmse = rmse

                            # Store the minimum RMSE and MAE in dictionaries
                            XGB_RMSE[selected_car] = min_rmse
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 2:
                        results, scaler = lr_cross_validate(vehicle_files, selected_car, save_dir=save_dir)

                        # Print overall results
                        if results:
                            min_rmse = float('inf')
                            for fold_num, rmse, mae in results:
                                print(f"Fold: {fold_num}, RMSE: {rmse}, MAE: {mae}")
                                if rmse < min_rmse:
                                    min_rmse = rmse

                            # Store the minimum RMSE and MAE in dictionaries
                            LR_RMSE[selected_car] = min_rmse
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if train_choice == 3:
                        results, scaler = svr_cross_validate(vehicle_files, selected_car, save_dir=save_dir)

                        # Print overall results
                        if results:
                            min_rmse = float('inf')
                            for fold_num, rmse, mae in results:
                                print(f"Fold: {fold_num}, RMSE: {rmse}, MAE: {mae}")
                                if rmse < min_rmse:
                                    min_rmse = rmse

                            # Store the minimum RMSE and MAE in dictionaries
                            SVR_RMSE[selected_car] = min_rmse
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")

                print(f"XGB RMSE: {XGB_RMSE}")
                print(f"LR RMSE: {LR_RMSE}")
                print(f"SVR RMSE: {SVR_RMSE}")

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
                print("5: Plotting Power Difference Graph")
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
                        if plot == 1:
                            plot_power(sample_files, selected_car, 'stacked')
                        elif plot == 2:
                            plot_power(sample_files, selected_car, 'model')
                        elif plot == 3:
                            plot_power(sample_files, selected_car, 'data')
                        elif plot == 4:
                            plot_power(sample_files, selected_car, 'comparison')
                        elif plot == 5:
                            plot_power(sample_files, selected_car, 'difference')
                        elif plot == 6:
                            plot_power(sample_files, selected_car, 'd_altitude')
                        elif plot == 7:
                            plot_energy(sample_files, selected_car, 'model')
                        elif plot == 8:
                            plot_energy(sample_files, selected_car, 'data')
                        elif plot == 9:
                            plot_energy(sample_files, selected_car, 'comparison')
                        elif plot == 10:
                            plot_energy(sample_files, selected_car, 'altitude')
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
                            plot_energy_dis(vehicle_files[selected_car], folder_path, selected_car, 'model')
                        elif plot == 6:
                            plot_energy_dis(vehicle_files[selected_car], folder_path, selected_car, 'data')
                        elif plot == 7:
                            plot_energy_dis(vehicle_files[selected_car], folder_path, selected_car, 'learning')

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
