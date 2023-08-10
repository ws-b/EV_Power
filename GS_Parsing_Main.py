import os
from GS_preprocessing import get_file_list, parse_spacebar, process_files_combined, process_files_trip_by_trip, process_files_all_trips


def pre_process():
    while True:
        print("1: Initial GS Data Parsing")
        print("2: Calculating Speed(m/s), Acceleration(m/s^2)")
        print("3: Trip by Trip Parsing")
        print("4: Parsing All trips in one file")
        print("5: Quitting the program.")
        choice = int(input("Enter number you want to run: "))

        if choice == 1:
            folder_path = os.path.normpath('D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터')
            save_path = os.path.normpath('D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\processed')
            file_lists = get_file_list(folder_path)
            parse_spacebar(file_lists, folder_path, save_path)
            break

        elif choice == 2:
            folder_path = os.path.normpath('D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\processed')
            save_path = os.path.normpath('D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\speed-acc')
            file_list = get_file_list(folder_path)
            process_files_combined(file_list, folder_path, save_path)
            break

        elif choice == 3:
            folder_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\speed-acc')
            save_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\trip_by_trip')
            file_list = get_file_list(folder_path)
            process_files_trip_by_trip(file_list, folder_path, save_path)
            break

        elif choice == 4:
            folder_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\speed-acc')
            save_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\1min_trip')
            file_list = get_file_list(folder_path)
            process_files_all_trips(file_list, folder_path, save_path)
            break

        elif choice == 5:
            print("Quitting the program.")
            return

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    pre_process()