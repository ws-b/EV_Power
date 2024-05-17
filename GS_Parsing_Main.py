import os
import platform
from GS_preprocessing import (get_file_list, process_device_folders,process_files_trip_by_trip,
                              process_bms_files, process_gps_files, process_bms_altitude_files,
                              merge_bms_gps)

device_vehicle_mapping = {
    **{device: 'NiroEV' for device in ['01241228149', '01241228151', '01241228153', '01241228154', '01241228155']},
    **{device: 'Bongo3EV' for device in ['01241228162', '01241228179', '01241248642', '01241248723', '01241248829']},
    **{device: 'Ionic5' for device in ['01241227999', '01241228003', '01241228005', '01241228007', '01241228009',
                                      '01241228014', '01241228016', '01241228020', '01241228024', '01241228025',
                                      '01241228026', '01241228030', '01241228037', '01241228044', '01241228046',
                                      '01241228047', '01241248780', '01241248782', '01241248790', '01241248811',
                                      '01241248815', '01241248817', '01241248820', '01241248827', '01241364543',
                                      '01241364560', '01241364570', '01241364581', '01241592867', '01241592868',
                                      '01241592878', '01241592896', '01241592907', '01241597801', '01241597802',
                                      '01241248919']},
    **{device: 'Ionic6' for device in ['01241248713', '01241592904', '01241597763', '01241597804']},
    **{device: 'KonaEV' for device in ['01241228102', '01241228122', '01241228123', '01241228156', '01241228197',
                                      '01241228203', '01241228204', '01241248726', '01241248727', '01241364621',
                                      '01241124056']},
    **{device: 'Porter2EV' for device in ['01241228144', '01241228160', '01241228177', '01241228188', '01241228192']},
    **{device: 'EV6' for device in ['01241225206', '01241228048', '01241228049', '01241228050', '01241228051',
                                    '01241228053', '01241228054', '01241228055', '01241228057', '01241228059',
                                    '01241228073', '01241228075', '01241228076', '01241228082', '01241228084',
                                    '01241228085', '01241228086', '01241228087', '01241228090', '01241228091',
                                    '01241228092', '01241228094', '01241228095', '01241228097', '01241228098',
                                    '01241228099', '01241228103', '01241228104', '01241228106', '01241228107',
                                    '01241228114', '01241228124', '01241228132', '01241228134', '01241248679',
                                    '01241248818', '01241248831', '01241248833', '01241248842', '01241248843',
                                    '01241248850', '01241248860', '01241248876', '01241248877', '01241248882',
                                    '01241248891', '01241248892', '01241248900', '01241248903', '01241248908',
                                    '01241248912', '01241248913', '01241248921', '01241248924', '01241248926',
                                    '01241248927', '01241248929', '01241248932', '01241248933', '01241248934',
                                    '01241321943', '01241321947', '01241364554', '01241364575', '01241364592',
                                    '01241364627', '01241364638', '01241364714']},
    **{device: 'GV60' for device in ['01241228108', '01241228130', '01241228131', '01241228136', '01241228137',
                                    '01241228138']}
}

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
            source_paths = r"D:\SamsungSTF\Data\GSmbiz\bms_altitude\bms"
            destination_root = r"D:\SamsungSTF\Data\GSmbiz\BMS_Data"

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
                    start_path = r'D:\SamsungSTF\Data\GSmbiz\BMS_Data'
                    save_path = r'D:\SamsungSTF\Processed_Data/Merged'
                    process_bms_files(start_path, save_path, device_vehicle_mapping)
                elif data_choice == 2:
                    start_path = r'D:\SamsungSTF\Data\GSmbiz\BMS_Data'
                    save_path = r'D:\SamsungSTF\Processed_Data/Merged'
                    process_gps_files(start_path, save_path)
                elif data_choice == 3:
                    start_path = r'D:\SamsungSTF\Data\GSmbiz\BMS_Data'
                    save_path = r'D:\SamsungSTF\Processed_Data/Merged'
                    process_bms_altitude_files(start_path, save_path, device_vehicle_mapping)
                elif data_choice == 4:
                    start_path = '/Volumes/Data/SamsungSTF/Data/GSmbiz/GSmbiz'
                    merge_bms_gps(start_path)
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