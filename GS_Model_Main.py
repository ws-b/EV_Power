import os
from GS_preprocessing_1 import get_file_list
from GS_Cal_Power import Vehicle, process_files_energy, select_vehicle
from GS_filtering_data import move_files
from GS_plot_line import plot_energy_comparison
from GS_plot_scatter import plot_scatter_all_trip, plot_scatter_tbt
from GS_plot_energy_distribution import plot_bms_energy_dis, plot_model_energy_dis

def main():
    print("1: Ioniq5")
    print("2: Kona_EV")
    print("3: Porter_EV")
    print("4: Quitting the program.")
    car = int(input("Select Car you want to calculate: "))

    if car == 1:
        EV = select_vehicle(car)
        vehicle_name = 'ioniq5'
    elif car == 2:
        EV = select_vehicle(car)
        vehicle_name = 'kona_ev'
    elif car == 3:
        EV = select_vehicle(car)
        vehicle_name = 'porter_ev'
    elif car == 4:
        print("Quitting the program.")
        return
    else:
        print("Invalid choice. Please try again.")

    folder_path = os.path.normpath('D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터')
    folder_path = os.path.join(folder_path, vehicle_name)
    file_list = get_file_list(folder_path)

    while True:
        print("1: Calculate Power(W) using Model")
        print("2: Filtering Data")
        print("3: Plotting Energy Graph(Scatter, Line")
        print("4: Plotting Energy Distribution")
        print("5: Plotting All Graph")
        print("6: Quitting the program.")
        choice = int(input("Enter number you want to run: "))

        if choice == 1:
            process_files_energy(file_list, folder_path, EV)
            break
        elif choice == 2:
            moved_path = os.path.join(folder_path, 'moved')
            if not os.path.exists(moved_path):
                os.makedirs(moved_path)
            file_list = get_file_list(folder_path)
            move_files(file_list, folder_path, moved_path)
            break
        elif choice == 3:
            while True:
                print("1: Plotting All trip's Energy Graph(Scatter)")
                print("2: Plotting each trip's Energy Graph(Scatter)")
                print("3: Plotting each trip's Energy Comparison Graph(Line)")
                print("4: Plotting All graph")
                print("5: Quitting the program.")
                plot = int(input("Enter number you want to run: "))
                if plot == 1:
                    plot_scatter_all_trip(file_list, folder_path)
                    break
                elif plot == 2:
                    plot_scatter_tbt(file_list, folder_path)
                    break
                elif plot == 3:
                    plot_energy_comparison(file_list, folder_path)
                    break
                elif plot == 4:
                    plot_scatter_all_trip(file_list, folder_path)
                    plot_energy_comparison(file_list, folder_path)
                    break
                elif plot == 5:
                    print("Quitting the program.")
                    return
            break
        elif choice == 4:
            while True:
                print("1: Plotting Model's Energy Distribution")
                print("2: Plotting BMS's Energy Distribution")
                print("3: Plotting All graph")
                print("4: Quitting the program.")
                plot = int(input("Enter number you want to run: "))
                if plot == 1:
                    plot_model_energy_dis(file_list, folder_path)
                    break
                elif plot == 2:
                    plot_bms_energy_dis(file_list, folder_path)
                    break
                elif plot == 3:
                    plot_model_energy_dis(file_lists, folder_path)
                    plot_bms_energy_dis(file_lists, folder_path)
                    break
                elif plot == 4:
                    print("Quitting the program.")
                    return
            break
        elif choice == 5:
            plot_scatter_all_trip(file_list, folder_path)
            plot_scatter_tbt(file_list, folder_path)
            plot_energy_comparison(file_list, folder_path)
            plot_model_energy_dis(file_list, folder_path)
            plot_bms_energy_dis(file_list, folder_path)
            break
        elif choice == 6:
            print("Quitting the program.")
            return
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
