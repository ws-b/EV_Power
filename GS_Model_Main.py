import os
import platform
import random
import pickle
from GS_preprocessing import load_data_by_vehicle
from GS_Merge_Power import process_files_power, select_vehicle
from GS_plot import plot_power, plot_energy, plot_energy_scatter, plot_power_scatter, plot_energy_dis, plot_driver_energy_scatter
from GS_vehicle_dict import vehicle_dict
from GS_Train_XGboost import cross_validate as xgb_cross_validate, add_predicted_power_column as xgb_add_predicted_power_column
from GS_Train_RF import cross_validate as rf_cross_validate, add_predicted_power_column as rf_add_predicted_power_column
from GS_Train_SVM import cross_validate as svm_cross_validate, add_predicted_power_column as svm_add_predicted_power_column

def main():
    while True:
        # Select car
        print("1: NiroEV")
        print("2: Ionic5")
        print("3: Ionic6")
        print("4: KonaEV")
        print("5: EV6")
        print("6: GV60")
        print("7: Bongo3EV")
        print("8: Porter2EV")
        print("10: Quitting the program.")
        car = int(input("Select Car you want to calculate: "))

        car_options = {
            1: 'NiroEV',
            2: 'Ionic5',
            3: 'Ionic6',
            4: 'KonaEV',
            5: 'EV6',
            6: 'GV60',
            7: 'Bongo3EV',
            8: 'Porter2EV',
        }

        if platform.system() == "Windows":
            folder_path = os.path.normpath(r'D:\SamsungSTF\Processed_Data')
        elif platform.system() == "Darwin":
            folder_path = os.path.normpath('/Users/wsong/Documents/삼성미래과제/Processed_data')
        else:
            print("Unknown system.")
            return

        folder_path = os.path.join(folder_path, 'TripByTrip')
        if car in car_options:
            selected_car = car_options[car]
            EV = select_vehicle(car)
            vehicle_files = load_data_by_vehicle(folder_path, vehicle_dict, selected_car)
            vehicle_file_lists = vehicle_files.get(selected_car, [])
            vehicle_file_lists.sort()

            if not vehicle_files:
                print(f"No files found for the selected vehicle: {selected_car}")

            print(f"YOUR CHOICE IS {selected_car}")
        elif car == 10:
            print("Quitting the program.")
            return
        else:
            print("Invalid choice. Please try again.")
            continue

        while True:
            print("1: Calculate Power(W)")
            print("2: Train Model")
            print("3: Predicted Power(W) using Trained Model")
            print("4: Plotting Graph (Power & Energy)")
            print("5: Plotting Graph (Scatter, Energy Distribution)")
            print("6: Return to previous menu")
            print("7: Quitting the program.")
            choice = int(input("Enter number you want to run: "))

            if choice == 1:
                process_files_power(vehicle_file_lists, folder_path, EV)
            elif choice == 2:
                save_dir = os.path.join(os.path.dirname(folder_path), 'Models')
                while True:
                    print("1: Train Model using XGBoost")
                    print("2: Train Model using Random Forest")
                    print("3: Train Model using SVM")
                    choice = int(input("Enter number you want to run: "))
                    if choice == 1:
                        results, scaler = xgb_cross_validate(vehicle_files, selected_car, save_dir=save_dir)

                        # Save the scaler
                        scaler_path = os.path.join(save_dir, f'XGB_scaler_{selected_car}.pkl')
                        with open(scaler_path, 'wb') as f:
                            pickle.dump(scaler, f)
                        print(f"Scaler saved at {scaler_path}")

                        # Print overall results
                        if results:
                            for fold_num, rmse, nrmse, percent_rmse in results:
                                print(f"Fold: {fold_num}, RMSE: {rmse}, NRMSE: {nrmse}, Percent RMSE: {percent_rmse}")
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if choice == 2:
                        results, scaler = rf_cross_validate(vehicle_files, selected_car, save_dir=save_dir)

                        # Save the scaler
                        scaler_path = os.path.join(save_dir, f'RF_scaler_{selected_car}.pkl')
                        with open(scaler_path, 'wb') as f:
                            pickle.dump(scaler, f)
                        print(f"Scaler saved at {scaler_path}")

                        # Print overall results
                        if results:
                            for fold_num, rmse, nrmse, percent_rmse in results:
                                print(f"Fold: {fold_num}, RMSE: {rmse}, NRMSE: {nrmse}, Percent RMSE: {percent_rmse}")
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    if choice == 3:
                        results, scaler = svm_cross_validate(vehicle_files, selected_car, save_dir=save_dir)

                        # Save the scaler
                        scaler_path = os.path.join(save_dir, f'SVM_scaler_{selected_car}.pkl')
                        with open(scaler_path, 'wb') as f:
                            pickle.dump(scaler, f)
                        print(f"Scaler saved at {scaler_path}")

                        # Print overall results
                        if results:
                            for fold_num, rmse, nrmse, percent_rmse in results:
                                print(f"Fold: {fold_num}, RMSE: {rmse}, NRMSE: {nrmse}, Percent RMSE: {percent_rmse}")
                        else:
                            print(f"No results for the selected vehicle: {selected_car}")
                    else:
                        print("Invalid choice. Please try again.")
                        continue
                    break

            elif choice == 3:
                while True:
                    print("1: XGBoost Model")
                    print("2: Random Forest Model")
                    print("3: SVM Model")
                    choice = int(input("Enter number you want to run: "))
                    if choice == 1:
                        model_path = os.path.join(os.path.dirname(folder_path), 'Models',
                                                  f'XGB_best_model_{selected_car}.json')
                        scaler_path = os.path.join(os.path.dirname(folder_path), 'Models',
                                                   f'XGB_scaler_{selected_car}.pkl')

                        if not vehicle_file_lists:
                            print(f"No files to process for the selected vehicle: {selected_car}")
                            return

                        # Load the scaler
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)

                        xgb_add_predicted_power_column(vehicle_file_lists, model_path, scaler)
                    elif choice == 2:
                        model_path = os.path.join(os.path.dirname(folder_path), 'Models',
                                                  f'RF_best_model_{selected_car}.json')
                        scaler_path = os.path.join(os.path.dirname(folder_path), 'Models',
                                                   f'RF_scaler_{selected_car}.pkl')

                        if not vehicle_file_lists:
                            print(f"No files to process for the selected vehicle: {selected_car}")
                            return

                        # Load the scaler
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)

                        rf_add_predicted_power_column(vehicle_file_lists, model_path, scaler)
                    elif choice == 3:
                        model_path = os.path.join(os.path.dirname(folder_path), 'Models',
                                                  f'SVM_best_model_{selected_car}.json')
                        scaler_path = os.path.join(os.path.dirname(folder_path), 'Models',
                                                   f'SVM_scaler_{selected_car}.pkl')

                        if not vehicle_file_lists:
                            print(f"No files to process for the selected vehicle: {selected_car}")
                            return

                        # Load the scaler
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)

                        svm_add_predicted_power_column(vehicle_file_lists, model_path, scaler)

                    else:
                        print("Invalid choice. Please try again.")
                        continue
                    break

            elif choice == 4:
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
                    print("11: Return to previous menu")
                    print("12: Quitting the program.")

                    selections = input("Enter the numbers you want to run, separated by commas (e.g., 1,2,3): ")
                    selections_list = selections.split(',')

                    for selection in selections_list:
                        plot = int(selection.strip())
                        if plot == 1:
                            plot_power(vehicle_file_lists, folder_path, 'stacked')
                        elif plot == 2:
                            plot_power(vehicle_file_lists, folder_path, 'model')
                        elif plot == 3:
                            plot_power(vehicle_file_lists, folder_path, 'data')
                        elif plot == 4:
                            plot_power(vehicle_file_lists, folder_path, 'comparison')
                        elif plot == 5:
                            plot_power(vehicle_file_lists, folder_path, 'difference')
                        elif plot == 6:
                            plot_power(vehicle_file_lists, folder_path, 'd_altitude')
                        elif plot == 7:
                            plot_energy(random.sample(vehicle_file_lists, 5), folder_path, 'model')
                        elif plot == 8:
                            plot_energy(random.sample(vehicle_file_lists, 5), folder_path, 'data')
                        elif plot == 9:
                            plot_energy(random.sample(vehicle_file_lists, 5), folder_path, 'comparison')
                        elif plot == 10:
                            plot_energy(vehicle_file_lists, folder_path, 'altitude')
                        elif plot == 11:
                            break
                        elif plot == 12:
                            print("Quitting the program.")
                            return
                        else:
                            print(f"Invalid choice: {plot}. Please try again.")
                    else:
                        continue
                    break
            elif choice == 5:
                while True:
                    print("1: Plotting Energy Scatter Graph")
                    print("2: Plotting Fitting Scatter Graph")
                    print("3: Plotting Individual Driver's Scatter Graph")
                    print("4: Plotting Power and Delta_altitude Graph")

                    print("5: Plotting Model Energy Distribution Graph")
                    print("6: Plotting Data Energy Distribution Graph")
                    print("7: Plotting Fitting Energy Distribution Graph")
                    print("8: ")
                    print("9: Return to previous menu.")
                    print("10: Quitting the program.")

                    selections = input("Enter the numbers you want to run, separated by commas (e.g., 1,2,3): ")
                    selections_list = selections.split(',')
                    for selection in selections_list:
                        plot = int(selection.strip())
                        if plot == 1:
                            plot_energy_scatter(vehicle_file_lists, folder_path, selected_car, 'model')
                        elif plot == 2:
                            plot_energy_scatter(vehicle_file_lists, folder_path, selected_car, 'fitting')
                        elif plot == 3:
                            sample_ids = random.sample(vehicle_dict[selected_car], 5)
                            for id in sample_ids:
                                sample_files = [f for f in vehicle_file_lists if id in f]
                                if len(sample_files) < 50:
                                    pass
                                else:
                                    plot_driver_energy_scatter(sample_files, folder_path, selected_car, id)
                        elif plot == 4:
                            plot_power_scatter(vehicle_file_lists, folder_path)
                        elif plot == 5:
                            plot_energy_dis(vehicle_file_lists, folder_path, selected_car, 'model')
                        elif plot == 6:
                            plot_energy_dis(vehicle_file_lists, folder_path, selected_car, 'data')
                        elif plot == 7:
                            plot_energy_dis(vehicle_file_lists, folder_path, selected_car, 'fitting')
                        elif plot == 8:
                            break
                        elif plot == 9:
                            break
                        elif plot == 10:
                            print("Quitting the program.")
                            return
                        else:
                            print(f"Invalid choice: {plot}. Please try again.")
                    else:
                        continue
                    break
            elif choice == 6:
                break
            elif choice == 7:
                print("Quitting the program.")
                return
            else:
                print("Invalid choice. Please try again.")
                continue

if __name__ == "__main__":
    main()
