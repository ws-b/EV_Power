import os
import platform
from GS_preprocessing import get_file_list, parse_spacebar, process_files_combined, process_files_trip_by_trip, merge_csv_files


def pre_process():
    while True:
        print("1: Initial GS Data Parsing")
        print("2: Calculating Speed(m/s), Acceleration(m/s^2)")
        print("3: Trip by Trip Parsing")
        print("6: Quitting the program.")
        choice = int(input("Enter number you want to run: "))

        if choice == 1:
            if platform.system() == "Windows":
                folder_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터')
                save_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\processed')
            elif platform.system() == "Darwin":
                folder_path = os.path.normpath('/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터')
                save_path = os.path.normpath('/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/processed')
            else:
                print("Unknown system.")
                return

            file_lists = get_file_list(folder_path)
            parse_spacebar(file_lists, folder_path, save_path)
            break

        elif choice == 2:
            if platform.system() == "Windows":
                folder_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\processed')
                save_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\speed-acc')
            elif platform.system() == "Darwin":
                folder_path = os.path.normpath('/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/processed')
                save_path = os.path.normpath('/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/speed-acc')
            else:
                print("Unknown system.")
                return

            file_list = get_file_list(folder_path)
            process_files_combined(file_list, folder_path, save_path)
            break

        elif choice == 3:
            if platform.system() == "Windows":
                folder_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\speed-acc')
                save_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\trip_by_trip')
            elif platform.system() == "Darwin":
                folder_path = os.path.normpath('/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/speed-acc')
                save_path = os.path.normpath('/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip')
            else:
                print("Unknown system.")
                return

            file_list = get_file_list(folder_path)
            process_files_trip_by_trip(file_list, folder_path, save_path)
            break

        elif choice == 4:
            if platform.system() == "Windows":
                folder_path = os.path.normpath(r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\trip_by_trip')
                save_path = os.path.normpath()
            elif platform.system() == "Darwin":
                folder_path = os.path.normpath('/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip')
                save_path = os.path.normpath('/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/merged')
            else:
                print("Unknown system.")
                return
            file_list = get_file_list(folder_path)
            merge_csv_files(file_list, folder_path,save_path)
            break

        elif choice == 6:
            print("Quitting the program.")
            return

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    pre_process()