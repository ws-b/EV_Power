import os
import platform
from GS_preprocessing import get_file_list
from GS_Merge_Power import process_files_power, select_vehicle
from GS_plot import plot_power, plot_energy, plot_energy_scatter, plot_power_scatter, plot_energy_dis
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
        print("3: Plotting Graph (Power & Energy)")
        print("4: Plotting Graph (Scatter, Energy Distribution)")
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
                    break
                elif choice == 2:
                    break
                elif choice == 9:
                    print("Quitting the program.")
                    return
                else:
                    print("Invalid choice. Please try again.")

            break
        elif choice == 3:
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
                print("11: Quitting the program.")
                plot = int(input("Enter number you want to run: "))
                if plot == 1:
                    plot_power(file_lists, folder_path, 'stacked')
                    break
                elif plot == 2:
                    plot_power(file_lists, folder_path, 'model')
                    break
                elif plot == 3:
                    plot_power(file_lists, folder_path, 'data')
                    break
                elif plot == 4:
                    plot_power(file_lists, folder_path, 'comparison')
                    break
                elif plot == 5:
                    plot_power(file_lists, folder_path, 'difference')
                    break
                elif plot == 6:
                    plot_power(file_lists, folder_path, 'd_altitude')
                    break
                elif plot == 7:
                    plot_energy(file_lists, folder_path, 'model')
                    break
                elif plot == 8:
                    plot_energy(file_lists, folder_path, 'data')
                    break
                elif plot == 9:
                    plot_energy(file_lists, folder_path, 'comparison')
                    break
                elif plot == 10:
                    plot_energy(file_lists, folder_path, 'altitude')
                    break
                elif plot == 11:
                    print("Quitting the program.")
                    return
            break
        elif choice == 4:
            while True:
                print("1: Plotting Energy Scatter Graph")
                print("2: Plotting Fitting Scatter Graph")
                print("3: Plotting Power and Delta_altitude Graph")

                print("5: Plotting Model Energy Distribution Graph")
                print("6: Plotting Data Energy Distribution Graph")
                print("7: Plotting Fitting Energy Distribution Graph")
                print("10: Quitting the program.")
                plot = int(input("Enter number you want to run: "))
                if plot == 1:
                    plot_energy_scatter(file_lists, folder_path, 'model')
                    break
                elif plot == 2:
                    plot_energy_scatter(file_lists, folder_path, 'fitting')
                    break
                elif plot == 3:
                    plot_power_scatter(file_lists, folder_path)
                    break
                elif plot == 5:
                    plot_energy_dis(file_lists, folder_path, 'model')
                    break
                elif plot == 6:
                    plot_energy_dis(file_lists, folder_path, 'data')
                    break
                elif plot == 7:
                    plot_energy_dis(file_lists, folder_path, 'fitting')
                    break
                elif plot == 10:
                    print("Quitting the program.")
                    return
            break
        elif choice == 6:
            print("Quitting the program.")
            return
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()