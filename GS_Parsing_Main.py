import os
import platform
from GS_preprocessing import (get_file_list, process_device_folders,
                              process_files_combined, process_files_trip_by_trip,
                              process_bms_files, process_gps_files, process_bms_altitude_files,
                              merge_bms_gps)

def pre_process():
    while True:
        print("1: Move device folders based on device number and type (New Files Only)")
        print("2: Process specific data type (BMS, GPS, Merge BMS and GPS)")
        print("3: Calculating Speed (m/s), Acceleration (m/s^2)")
        print("4: Trip by Trip Parsing")
        print("5: ")
        print("6: Quitting the program.")
        choice = input("Enter the number you want to run: ")

        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        choice = int(choice)

        if choice == 1:
            source_paths = [
                '/Volumes/Data/SamsungSTF/Data/GSmbiz/bms',
                '/Volumes/Data/SamsungSTF/Data/GSmbiz/bms_altitude'
            ]
            destination_root = '/Volumes/Data/SamsungSTF/Data/GSmbiz/bms_gps_data'

            process_device_folders(source_paths, destination_root)
            break

        elif choice == 2:
            print("Select the data type to process:")
            print("1: Process BMS data only")
            print("2: Process GPS data only")
            print("3: Process BMS and Altitude data")
            print("4: Merge BMS and GPS data")
            data_choice = input("Enter your choice: ")

            if data_choice.isdigit() and int(data_choice) in [1, 2, 3, 4]:
                data_choice = int(data_choice)
                if data_choice == 1:
                    start_path = '/Users/wsong/Downloads/testcase/bms_gps_data'
                    process_bms_files(start_path)
                elif data_choice == 2:
                    start_path = '/Users/wsong/Downloads/testcase/bms_gps_data'
                    process_gps_files(start_path)
                elif data_choice == 3:
                    start_path = '/Users/wsong/Downloads/testcase/bms_gps_data'
                    process_bms_altitude_files(start_path)
                elif data_choice == 4:
                    start_path = '/Users/wsong/Downloads/testcase/bms_gps_data'
                    merge_bms_gps(start_path)
            else:
                print("Invalid choice for data processing.")

        elif choice == 3:
            if platform.system() == "Windows":
                folder_path = os.path.normpath('')
                save_path = os.path.normpath('')
            elif platform.system() == "Darwin":
                folder_path = os.path.normpath(
                    '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/processed')
                save_path = os.path.normpath(
                    '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/speed-acc')
            else:
                print("Unknown system.")
                return

            file_list = get_file_list(folder_path)
            process_files_combined(file_list, folder_path, save_path)
            break

        elif choice == 4:
            if platform.system() == "Windows":
                folder_path = os.path.normpath('')
                save_path = os.path.normpath('')
            elif platform.system() == "Darwin":
                folder_path = os.path.normpath(
                    '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/speed-acc')
                save_path = os.path.normpath(
                    '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip')
            else:
                print("Unknown system.")
                return

            file_list = get_file_list(folder_path)
            process_files_trip_by_trip(file_list, folder_path, save_path)
            break

        elif choice == 6:
            print("Quitting the program.")
            return

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    pre_process()