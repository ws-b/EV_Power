import os
import platform
from GS_preprocessing import get_file_list
from GS_Merge_Power import process_files_power, select_vehicle
from GS_plot import plot_power, plot_energy, plot_energy_scatter, plot_power_scatter, plot_energy_dis
from GS_vehicle_dict import vehicle_dict
from GS_Train import cross_validate, load_data_by_vehicle, add_predicted_power_column

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
            all_file_lists = get_file_list(folder_path)
            file_lists = [file for file in all_file_lists if any(vehicle_id in file for vehicle_id in vehicle_dict[selected_car])]
            print(f"YOUR CHOICE IS {selected_car}")
        elif car == 10:
            print("Quitting the program.")
            return
        else:
            print("Invalid choice. Please try again.")
            continue

        file_lists.sort()

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
                process_files_power(file_lists, folder_path, EV)
            elif choice == 2:
                base_dir = folder_path
                save_dir = os.path.join(os.path.dirname(folder_path), 'Models')

                vehicle_files = load_data_by_vehicle(base_dir, vehicle_dict, selected_car)
                if not vehicle_files:
                    print(f"No files found for the selected vehicle: {selected_car}")
                    return

                results = cross_validate(vehicle_files, selected_car, save_dir=save_dir)

                # Print overall results
                if results:
                    for fold_num, rmse in results:
                        print(f"Fold: {fold_num}, RMSE: {rmse}")
                else:
                    print(f"No results for the selected vehicle: {selected_car}")
            elif choice == 3:
                base_dir = folder_path
                model_path = os.path.join(os.path.dirname(folder_path), 'Models', f'best_model_{selected_car}.json')

                vehicle_files = load_data_by_vehicle(base_dir, vehicle_dict, selected_car)
                if not vehicle_files:
                    print(f"No files found for the selected vehicle: {selected_car}")
                    return

                files = vehicle_files.get(selected_car, [])
                if not files:
                    print(f"No files to process for the selected vehicle: {selected_car}")
                    return

                add_predicted_power_column(files, model_path)
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

                    selections = input("Enter the numbers you want to run, separated by commas (e.g., 1,2,3): ")
                    selections_list = selections.split(',')

                    for selection in selections_list:
                        plot = int(selection.strip())
                        if plot == 1:
                            plot_power(file_lists, folder_path, 'stacked')
                        elif plot == 2:
                            plot_power(file_lists, folder_path, 'model')
                        elif plot == 3:
                            plot_power(file_lists, folder_path, 'data')
                        elif plot == 4:
                            plot_power(file_lists, folder_path, 'comparison')
                        elif plot == 5:
                            plot_power(file_lists, folder_path, 'difference')
                        elif plot == 6:
                            plot_power(file_lists, folder_path, 'd_altitude')
                        elif plot == 7:
                            plot_energy(file_lists, folder_path, 'model')
                        elif plot == 8:
                            plot_energy(file_lists, folder_path, 'data')
                        elif plot == 9:
                            plot_energy(file_lists, folder_path, 'comparison')
                        elif plot == 10:
                            plot_energy(file_lists, folder_path, 'altitude')
                        elif plot == 11:
                            break
                        else:
                            print(f"Invalid choice: {plot}. Please try again.")
                    else:
                        # continue the inner loop if break wasn't hit
                        continue
                    # break the inner loop if break was hit
                    break
            elif choice == 5:
                while True:
                    print("1: Plotting Energy Scatter Graph")
                    print("2: Plotting Fitting Scatter Graph")
                    print("3: Plotting Power and Delta_altitude Graph")
                    print("4: Return to previous menu")
                    print("5: Plotting Model Energy Distribution Graph")
                    print("6: Plotting Data Energy Distribution Graph")
                    print("7: Plotting Fitting Energy Distribution Graph")
                    print("10: Quitting the program.")

                    selections = input("Enter the numbers you want to run, separated by commas (e.g., 1,2,3): ")
                    selections_list = selections.split(',')
                    for selection in selections_list:
                        plot = int(selection.strip())
                        if plot == 1:
                            plot_energy_scatter(file_lists, folder_path, 'model')
                        elif plot == 2:
                            plot_energy_scatter(file_lists, folder_path, 'fitting')
                        elif plot == 3:
                            plot_power_scatter(file_lists, folder_path)
                        elif plot == 4:
                            break
                        elif plot == 5:
                            plot_energy_dis(file_lists, folder_path, 'model')
                        elif plot == 6:
                            plot_energy_dis(file_lists, folder_path, 'data')
                        elif plot == 7:
                            plot_energy_dis(file_lists, folder_path, 'fitting')
                        elif plot == 10:
                            print("Quitting the program.")
                            return
                        else:
                            print(f"Invalid choice: {plot}. Please try again.")
                    else:
                        # continue the inner loop if break wasn't hit
                        continue
                    # break the inner loop if break was hit
                    break
            elif choice == 6:
                break
            elif choice == 7:
                print("Quitting the program.")
                return
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
