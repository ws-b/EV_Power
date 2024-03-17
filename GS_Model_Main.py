import os
import platform
from GS_preprocessing import get_file_list
from GS_Merge_Power import process_files_power, select_vehicle
from GS_plot import plot_power, plot_energy, plot_energy_scatter, plot_energy_dis
from GS_Fitting import fitting

def main():
    print("1: Ioniq5")
    print("2: Kona_EV")
    print("3: Ioniq5_2")
    print("4: Quitting the program.")
    car = int(input("Select Car you want to calculate: "))
    if platform.system() == "Windows":
        folder_path = os.path.normpath('D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터')
    elif platform.system() == "Darwin":
        folder_path = os.path.normpath('/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터')
    else:
        print("Unknown system.")
        return
    folder_path = os.path.join(folder_path, 'trip_by_trip')

    if car == 1: #ioniq5
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if '01241248782' in file]
    elif car == 2: #kona_ev
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if '01241248726' in file]
    elif car == 3: #porter_ev
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if '01241592868' in file]
    elif car == 4:
        print("Quitting the program.")
        return
    else:
        print("Invalid choice. Please try again.")

    file_lists.sort()

    while True:
        print("1: Calculate Power(W) using Model & Filtering Data")
        print("2: Fitting Model")
        print("3: Plotting Energy Distribution")
        print("4: Plotting All Graph")
        print("5: Plotting Energy Graph(Scatter, Line)")
        print("6: Quitting the program.")
        choice = int(input("Enter number you want to run: "))

        if choice == 1:
            process_files_power(file_lists, folder_path, EV)
            break
        elif choice == 2:
            while True:
                print("1: Fitting Model")
                print("2: Plotting Energy Distribution")
                print("3: Plotting Energy/Power Comparison Graph")
                print("4: Plotting Scatter Graph")
                print("5: Plotting contour Graph ")
                print("6: Plotting ")
                print("7: Plotting ")
                print("8: Plotting ")
                print("9: Quitting the program.")
                choice = int(input("Enter number you want to run: "))

                if choice == 1:
                    fitting(file_lists, folder_path)
                    plot_fit_scatter_all_trip(file_lists, folder_path)
                    break
                elif choice == 2:
                    plot_fit_model_energy_dis(file_lists, folder_path)
                    break
                elif choice == 3:
                    plot_fit_energy_comparison(file_lists, folder_path)
                    plot_fit_power_comparison(file_lists, folder_path)
                    break
                elif choice == 4:
                    plot_fit_scatter_all_trip(file_lists, folder_path)
                    break
                elif choice == 5:
                    break
                elif choice == 6:
                    break
                elif choice == 7:
                    break
                elif choice == 8:
                    break
                elif choice == 9:
                    print("Quitting the program.")
                    return
                else:
                    print("Invalid choice. Please try again.")

            break
        elif choice == 3:
            while True:
                print("1: Plotting Model's Energy Distribution")
                print("2: Plotting Data's Energy Distribution")
                print("3: Plotting All graph")
                print("4: Quitting the program.")
                plot = int(input("Enter number you want to run: "))
                if plot == 1:
                    plot_model_energy_dis(file_lists, folder_path)
                    break
                elif plot == 2:
                    plot_bms_energy_dis(file_lists, folder_path)
                    break
                elif plot == 3:
                    plot_model_energy_dis(file_lists, folder_path)
                    plot_bms_energy_dis(file_lists, folder_path)
                    break
                elif plot == 4:
                    print("Quitting the program.")
                    return
            break
        elif choice == 4:
            plot_scatter_all_trip(file_lists, folder_path)
            energy_scatter(file_lists, folder_path, 'model')
            plot_scatter_tbt(file_lists, folder_path)
            plot_energy_comparison(file_lists, folder_path)
            plot_model_energy_dis(file_lists, folder_path)
            plot_bms_energy_dis(file_lists, folder_path)
            break
        elif choice == 5:
            while True:
                print("1: Plotting Stacked Power Plot Term by Term")
                print("2: Plotting each trip's Energy Graph(Line)")
                print("3: Plotting each trip's Energy Comparison Graph(Line)")

                print("18: Quitting the program.")
                plot = int(input("Enter number you want to run: "))
                if plot == 1:
                    plot_energy(file_lists, folder_path, 'stacked')
                    break
                elif plot == 2:
                    while True:
                        print("1: Plotting Model's Energy Graph")
                        print("2: Plotting Data's Energy Graph")
                        print("3: Plotting All graph")
                        print("4: Quitting the program.")
                        plot = int(input("Enter number you want to run: "))
                        if plot == 1:
                            plot_energy(file_lists, folder_path, 'model')
                            break
                        elif plot ==2:
                            plot_energy(file_lists, folder_path, 'data')
                            break
                        elif plot == 3:
                            plot_energy(file_lists, folder_path, 'model')
                            plot_energy(file_lists, folder_path, 'data')
                            break
                        elif plot == 4:
                            print("Quitting the program.")
                            return
                    break
                elif plot == 3:
                    plot_energy(file_lists, folder_path, 'comparison')
                    break
                elif plot == 4:
                    plot_speed_power(file_lists, folder_path)
                    break
                elif plot == 5:
                    plot_power_comparison(file_lists, folder_path)
                    break
                elif plot == 6:
                    plot_power_diff(file_lists, folder_path)
                    break
                elif plot == 7:
                    plot_alt_energy_comparison(file_lists, folder_path)
                    break
                elif plot == 8:
                    plot_correlation(file_lists, folder_path)
                    break
                elif plot == 9:
                    plot_scatter_all_trip(file_lists, folder_path)
                    break
                elif plot == 10:
                    plot_scatter_tbt(file_lists, folder_path)
                    break
                elif plot == 11:
                    plot_temp_energy(file_lists, folder_path)
                    break
                elif plot == 12:
                    plot_distance_energy(file_lists, folder_path)
                    break
                elif plot == 13:
                    plot_temp_energy_wh_mile(file_lists, folder_path)
                    break
                elif plot == 14:
                    plot_energy_temp_speed(file_lists, folder_path)
                    break
                elif plot == 15:
                    plot_energy_temp_speed_3d(file_lists, folder_path)
                    break
                elif plot == 16:
                    plot_energy_temp_speed_normalized(file_lists, folder_path)
                    break
                elif plot == 17:
                    plot_bms_energy_pdf(file_lists, folder_path, EV)
                    break
                elif plot == 18:
                    print("Quitting the program.")
                    return
                else:
                    print("Invalid choice. Please try again.")
            break
        elif choice == 6:
            print("Quitting the program.")
            return
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()