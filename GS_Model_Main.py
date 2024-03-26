import os
import platform
from GS_preprocessing import get_file_list
from GS_Merge_Power import process_files_power, select_vehicle
from GS_plot import plot_power, plot_energy, plot_energy_scatter, plot_power_scatter, plot_energy_dis
from GS_Fitting import fitting
def main():
    niroEV = ['01241228149', '01241228151', '01241228153', '01241228154', '01241228155']
    bongo3EV = ['01241228162', '01241228179', '01241248642', '01241248723', '01241248829']
    ionic5 = ['01241227999', '01241228003', '01241228005', '01241228007', '01241228009', '01241228014', '01241228016',
              '01241228020', '01241228024', '01241228025', '01241228026', '01241228030', '01241228037', '01241228044',
              '01241228046', '01241228047', '01241248780', '01241248782', '01241248790', '01241248811', '01241248815',
              '01241248817', '01241248820', '01241248827', '01241364543', '01241364560', '01241364570', '01241364581',
              '01241592867', '01241592868', '01241592878', '01241592896', '01241592907', '01241597801', '01241597802']
    ionic6 = ['01241248713', '01241592904', '01241597763', '01241597804']
    konaEV = ['01241228102', '01241228122', '01241228123', '01241228156', '01241228197', '01241228203', '01241228204',
              '01241248726', '01241248727', '01241364621']
    porter2EV = ['01241228144', '01241228160', '01241228177', '01241228188', '01241228192']
    EV6 = ['01241225206', '01241228048', '01241228049', '01241228050', '01241228051', '01241228053', '01241228054',
           '01241228055', '01241228057', '01241228059', '01241228073', '01241228075', '01241228076', '01241228082',
           '01241228084', '01241228085', '01241228086', '01241228087', '01241228090', '01241228091', '01241228092',
           '01241228094', '01241228095', '01241228097', '01241228098', '01241228099', '01241228103', '01241228104',
           '01241228106', '01241228107', '01241228114', '01241228124', '01241228132', '01241228134', '01241248679',
           '01241248818', '01241248831', '01241248833', '01241248842', '01241248843', '01241248850', '01241248860',
           '01241248876', '01241248877', '01241248882', '01241248891', '01241248892', '01241248900', '01241248903',
           '01241248908', '01241248912', '01241248913', '01241248921', '01241248924', '01241248926', '01241248927',
           '01241248929', '01241248932', '01241248933', '01241248934', '01241321943', '01241321947', '01241364554',
           '01241364575', '01241364592', '01241364627', '01241364638', '01241364714']
    GV60 = ['01241228108', '01241228130', '01241228131', '01241228136', '01241228137', '01241228138']

    # Select car
    print("1: 니로EV")
    print("2: 봉고3EV")
    print("3: 아이오닉5")
    print("4: 아이오닉6")
    print("5: 코나EV")
    print("6: 포터2EV")
    print("7: EV6")
    print("8: GV60")
    print("10: Quitting the program.")
    car = int(input("Select Car you want to calculate: "))
    if platform.system() == "Windows":
        folder_path = os.path.normpath('')
    elif platform.system() == "Darwin":
        folder_path = os.path.normpath('/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터')
    else:
        print("Unknown system.")
        return
    folder_path = os.path.join(folder_path, 'trip_by_trip')
    if car == 1:  # niroEV
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if any(niro_id in file for niro_id in niroEV)]
    elif car == 2:  # bongo3EV
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if any(bongo3_id in file for bongo3_id in bongo3EV)]
    elif car == 3:  # ionic5
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if any(ionic5_id in file for ionic5_id in ionic5)]
    elif car == 4: #ionic6
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if any(ionic6_id in file for ionic6_id in ionic6)]
    elif car == 5: # konaEV
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if any(kona_id in file for kona_id in konaEV)]
    elif car == 6: # porter2EV
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if any(porter2_id in file for porter2_id in porter2EV)]
    elif car == 7: # EV6
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if any(ev6_id in file for ev6_id in EV6)]
    elif car == 8: # GV60
        EV = select_vehicle(car)
        all_file_lists = get_file_list(folder_path)
        file_lists = [file for file in all_file_lists if any(gv60_id in file for gv60_id in GV60)]
    elif car == 9:
        return
    elif car == 10:
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

                selections = input("Enter the numbers you want to run, separated by commas (e.g., 1,2,3): ")
                selections_list = selections.split(',')

                for selection in selections_list:
                    plot = int(selection.strip())
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
                    else:
                        print(f"Invalid choice: {plot}. Please try again.")
            break
        elif choice == 4:
            while True:
                print("1: Plotting Energy Scatter Graph")
                print("2: Plotting Fitting Scatter Graph")
                print("3: Plotting Power and Delta_altitude Graph")
                print("4: Plotting ")
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
            break
        elif choice == 6:
            print("Quitting the program.")
            return
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()