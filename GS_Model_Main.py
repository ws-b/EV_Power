import os
import platform
from GS_preprocessing import get_file_list
from GS_Merge_Power import process_files_power, select_vehicle
from GS_plot_line import (
plot_energy_comparison,
plot_stacked_graph,
plot_model_energy,
plot_bms_energy,
plot_speed_power,
plot_power_comparison,
plot_power_diff,
plot_correlation,
plot_power_comparison_enlarge,
plot_fit_energy_comparison,
plot_fit_power_comparison
)
from GS_plot_scatter import (
plot_scatter_all_trip,
plot_scatter_tbt,
plot_temp_energy,
plot_distance_energy,
plot_temp_energy_wh_mile,
plot_energy_temp_speed,
plot_energy_temp_speed_3d,
plot_energy_temp_speed_normalized,
plot_fit_scatter_all_trip,
plot_fit_scatter_tbt
)
from GS_plot_energy_distribution import plot_bms_energy_dis, plot_model_energy_dis, plot_fit_model_energy_dis
from GS_Fitting import fitting, visualize_all_files
from GS_plot_contour import plot_contour

def main():
    print("1: Ioniq5")
    print("2: Kona_EV")
    print("3: Porter_EV")
    print("4: Quitting the program.")
    car = int(input("Select Car you want to calculate: "))
    if platform.system() == "Windows":
        folder_path = os.path.normpath('D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터')
    elif platform.system() == "Darwin":
        folder_path = os.path.normpath('/Users/wsong/Documents/KENTECH/삼성미래과제/한국에너지공과대학교_샘플데이터')
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
        file_lists = [file for file in all_file_lists if '01241228177' in file]
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
                    visualize_all_files(file_lists, folder_path)
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
                    #plot_fit_scatter_tbt(file_lists, folder_path)
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
                print("2: Plotting BMS's Energy Distribution")
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
            plot_scatter_tbt(file_lists, folder_path)
            plot_energy_comparison(file_lists, folder_path)
            plot_model_energy_dis(file_lists, folder_path)
            plot_bms_energy_dis(file_lists, folder_path)
            break
        elif choice == 5:
            while True:
                print("1: Plotting Power Plot Term by Term")
                print("2: Plotting each trip's Energy Graph(Line)")
                print("3: Plotting each trip's Energy Comparison Graph(Line)")
                print("4: Plotting Speed & Power Graph(Line)")
                print("5: Plotting each trip's Power Comparison Graph(Line)")
                print("6: Plotting each trip's Power difference Graph(Line)")
                print("7: Plotting each trip's Enlarged Power Comparison Graph(Line)")
                print("8: Plotting Correlation Graph")
                print("9: Plotting All trip's Energy Graph(Scatter)")
                print("10: Plotting each trip's Energy Graph(Scatter)")
                print("11: Plotting Temperature & Energy Graph(Scatter)")
                print("12: Plotting Distance & Energy Graph(Scatter)")
                print("13: Plotting Temperature & Energy Graph (Wh/mile) (Scatter)")
                print("14: Plotting Temperature & Energy & Speed Graph (Scatter)")
                print("15: Plotting Temperature & Energy & Speed Graph 3D (Scatter) ")
                print("16: Plotting Noramalized Temperature & Energy & Speed Graph")
                print("17: Plotting All figure")
                print("18: Quitting the program.")
                plot = int(input("Enter number you want to run: "))
                if plot == 1:
                    plot_stacked_graph(file_lists, folder_path)
                    break
                elif plot == 2:
                    while True:
                        print("1: Plotting Model's Energy Graph")
                        print("2: Plotting BMS's Energy Graph")
                        print("3: Plotting All graph")
                        print("4: Quitting the program.")
                        plot = int(input("Enter number you want to run: "))
                        if plot == 1:
                            plot_model_energy(file_lists, folder_path)
                            break
                        elif plot ==2:
                            plot_bms_energy(file_lists, folder_path)
                            break
                        elif plot == 3:
                            plot_model_energy(file_lists, folder_path)
                            plot_bms_energy(file_lists, folder_path)
                            break
                        elif plot == 4:
                            print("Quitting the program.")
                            return
                    break
                elif plot == 3:
                    plot_energy_comparison(file_lists, folder_path)
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
                    plot_power_comparison_enlarge(file_lists, folder_path, 0, 10)
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