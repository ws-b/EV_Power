import os
import platform
from GS_preprocessing import (get_file_list, process_device_folders,process_files_trip_by_trip,
                              process_bms_files, process_bms_altitude_files)
from GS_vehicle_dict import vehicle_dict

device_vehicle_mapping = {device: model for model, devices in vehicle_dict.items() for device in devices}

def pre_process():
    while True:
        print("1: Move device folders based on device number and type (New Files Only)")
        print("2: Process specific data type (BMS, GPS, Merge BMS and GPS)")
        print("3: Trip by Trip Parsing")
        print("4: Quitting the program.")
        choice = input("Enter the number you want to run: ")

        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        choice = int(choice)

        if choice == 1:
            source_paths = r"C:\Users\BSL\Desktop\bms"
            destination_root = r"D:\SamsungSTF\Data\GSmbiz\BMS_Data"

            process_device_folders(source_paths, destination_root)
            break

        elif choice == 2:
            print("Select the data type to process:")
            print("1: Process BMS data only")
            print("2: Process BMS and Altitude data")

            data_choice = input("Enter your choice: ")

            if data_choice.isdigit() and int(data_choice) in [1, 2, 3, 4]:
                data_choice = int(data_choice)
                if data_choice == 1:
                    start_path = r'D:\SamsungSTF\Data\GSmbiz\BMS_Data'
                    save_path = r'D:\SamsungSTF\Processed_Data/Merged'
                    process_bms_files(start_path, save_path, device_vehicle_mapping)
                elif data_choice == 2:
                    start_path = r'D:\SamsungSTF\Data\GSmbiz\BMS_Data'
                    save_path = r'D:\SamsungSTF\Processed_Data/Merged'
                    process_bms_altitude_files(start_path, save_path, device_vehicle_mapping)
            else:
                print("Invalid choice for data processing.")
            break

        elif choice == 3:
            if platform.system() == "Windows":
                start_path = os.path.normpath(r'D:\SamsungSTF\Processed_Data\Merged')
                save_path = os.path.normpath(r'D:\SamsungSTF\Processed_Data\TripByTrip')
            elif platform.system() == "Darwin":
                start_path = os.path.normpath(
                    '/Users/wsong/Documents/삼성미래과제/Processed_data/Merged')
                save_path = os.path.normpath(
                    '/Users/wsong/Documents/삼성미래과제/Processed_data/TripByTrip')
            else:
                print("Unknown system.")
                return

            file_list = get_file_list(start_path)
            process_files_trip_by_trip(file_list, start_path, save_path)
            break

        elif choice == 4:
            print("Quitting the program.")
            return

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    pre_process()