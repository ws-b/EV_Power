import os
import platform
from GS_preprocessing import (process_device_folders,process_files_trip_by_trip, process_files)
from GS_vehicle_dict import vehicle_dict

vehicle_type = {device: model for model, devices in vehicle_dict.items() for device in devices}

def pre_process():
    while True:
        print("1: Move device folders based on device number and type (New Files Only)")
        print("2: Pre-Process BMS Files")
        print("3: Trip by Trip Parsing")
        print("4: Return to previous menu")
        print("5: Quitting the program.")
        choice = input("Enter the number you want to run: ")

        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        choice = int(choice)

        if choice == 1:
            source_paths = r"C:\Users\WSONG\Desktop\drive-download-20240711T052431Z-001"
            destination_root = r"D:\SamsungSTF\Data\GSmbiz\BMS_Data"
            process_device_folders(source_paths, destination_root, False)
            process_device_folders(source_paths, destination_root, True)
            break

        elif choice == 2:
            start_path = r'D:\SamsungSTF\Data\GSmbiz\BMS_Data'
            save_path = r'D:\SamsungSTF\Processed_Data/Merged'
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
                
            process_files(start_path, save_path, vehicle_type, altitude=False)
            process_files(start_path, save_path, vehicle_type, altitude=True)
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
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            process_files_trip_by_trip(start_path, save_path)
            break

        elif choice == 4:
            break
        elif choice == 5:
            print("Quitting the program.")
            return

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    pre_process()