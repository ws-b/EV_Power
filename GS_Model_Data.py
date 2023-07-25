import os
from GS_preprocessing_1 import get_file_list
from GS_Cal_Power import Vehicle, process_files_energy, select_vehicle
from GS_plot_line import plot_energy_comparison

def main():
    while True:
        print("1: Ioniq5")
        print("2: Kona_EV")
        print("3: Porter_EV")
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
        else:
            print("Invalid choice. Please try again.")
            continue

        folder_path = os.path.normpath('D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터')
        folder_path = os.path.join(folder_path, vehicle_name)
        file_list = get_file_list(folder_path)

        while True:
            print("1: Calculate Energy(kWh) using Model")
            print("2: Plotting BMS_Energy(kWh), Model_Energy(kWh)")
            print("3: Plotting Energy Distribution")
            print("4: Quitting the program.")
            choice = int(input("Enter number you want to run: "))

            if choice == 1:
                process_files_energy(file_list, folder_path, EV)
                break
            elif choice == 2:
                plot_energy_comparison(file_list, folder_path)
                break
            elif choice == 3:

                break
            elif choice == 4:
                print("Quitting the program.")
                return
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
