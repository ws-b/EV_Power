import os
import platform
from GS_preprocessing import process_device_folders, process_files_trip_by_trip, process_files, delete_zero_kb_files
from GS_preprocessing_2 import merge_bms_data_by_device, process_files_trip_by_trip as process_trip_by_trip_soc
from GS_vehicle_dict import vehicle_dict

# OS에 따라 경로 다르게 설정하는 함수
def get_paths():
    if platform.system() == "Windows":
        base_source = r"C:\Users\BSL\Desktop\새 폴더"
        base_destination = r"D:\SamsungSTF\Data\GSmbiz\BMS_Data"
        processed_merged = r'D:\SamsungSTF\Processed_Data\Merged'
        processed_trip = r'D:\SamsungSTF\Processed_Data\TripByTrip'
        processed_trip_soc = r'D:\SamsungSTF\Processed_Data\TripByTrip_soc'
        processed_merged_period = r'D:\SamsungSTF\Processed_Data\Merged_period'
    else:  # Linux (WSL2)
        base_source = "/mnt/c/Users/BSL/Desktop/새 폴더"
        base_destination = "/mnt/d/SamsungSTF/Data/GSmbiz/BMS_Data"
        processed_merged = '/mnt/d/SamsungSTF/Processed_Data/Merged'
        processed_trip = '/mnt/d/SamsungSTF/Processed_Data/TripByTrip'
        processed_trip_soc = '/mnt/d/SamsungSTF/Processed_Data/TripByTrip_soc'
        processed_merged_period = '/mnt/d/SamsungSTF/Processed_Data/Merged_period'

    return base_source, base_destination, processed_merged, processed_trip, processed_trip_soc, processed_merged_period

def pre_process():
    base_source, base_destination, processed_merged, processed_trip, processed_trip_soc, processed_merged_period = get_paths()

    while True:
        print("1: Move device folders based on device number and type (New Files Only)")
        print("2: Pre-Process BMS Files")
        print("3: Trip by Trip Parsing")
        print("4: Trip by Trip (SOC 추정용) Parsing")
        print("5: Pre-Process BMS Files(년 단위 묶음), N days Parsing")
        print("6: Return to previous menu")
        print("7: Quitting the program.")
        choice = input("Enter the number you want to run: ")

        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        choice = int(choice)

        if choice == 1:
            process_device_folders(base_source, base_destination, False)
            process_device_folders(base_source, base_destination, True)
            delete_zero_kb_files(base_destination)
            break

        elif choice == 2:
            if not os.path.exists(processed_merged):
                os.makedirs(processed_merged, exist_ok=True)
                
            process_files(base_destination, processed_merged, vehicle_dict, altitude=False)
            process_files(base_destination, processed_merged, vehicle_dict, altitude=True)
            break

        elif choice == 3:
            if not os.path.exists(processed_trip):
                os.makedirs(processed_trip, exist_ok=True)
            process_files_trip_by_trip(processed_merged, processed_trip)
            break

        elif choice == 4:
            if not os.path.exists(processed_trip_soc):
                os.makedirs(processed_trip_soc, exist_ok=True)
            process_trip_by_trip_soc(processed_merged, processed_trip_soc)
            break

        elif choice == 5:
            if not os.path.exists(processed_merged_period):
                os.makedirs(processed_merged_period, exist_ok=True)
                
            merge_bms_data_by_device(base_destination, processed_merged_period, vehicle_dict, altitude=False, period_days=7)
            merge_bms_data_by_device(base_destination, processed_merged_period, vehicle_dict, altitude=True, period_days=7)
            break

        elif choice == 6:
            break

        elif choice == 7:
            print("Quitting the program.")
            return

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    pre_process()
