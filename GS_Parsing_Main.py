import os
from GS_preprocessing import process_device_folders,process_files_trip_by_trip, process_files, delete_zero_kb_files
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
            source_paths = r"C:\Users\BSL\Desktop\새 폴더" # 옮길 파일이 들어있는 폴더
            destination_root = r"D:\SamsungSTF\Data\GSmbiz\BMS_Data" # 옮길 폴더
            process_device_folders(source_paths, destination_root, False)
            process_device_folders(source_paths, destination_root, True)
            delete_zero_kb_files(destination_root)
            break

        elif choice == 2:
            start_path = r'D:\SamsungSTF\Data\GSmbiz\BMS_Data' # Raw Data 폴더
            save_path = r'D:\SamsungSTF\Processed_Data/Merged'# 월별 데이터 병합 폴더
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
                
            process_files(start_path, save_path, vehicle_type, altitude=False)
            process_files(start_path, save_path, vehicle_type, altitude=True)
            break

        elif choice == 3:
            start_path = os.path.normpath(r'D:\SamsungSTF\Processed_Data\Merged')
            save_path = os.path.normpath(r'D:\SamsungSTF\Processed_Data\TripByTrip')
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