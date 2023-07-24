import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from GS_preprocessing_1 import get_file_list


def main():
    while True:
        print("1: Calculate Energy(kWh) using Model")
        print("2: Data Filtering")
        print("3: Plotting Energy(kWh)")
        choice = int(input("Enter number you want to run: "))

        if choice == 1:
            folder_path = os.path.normpath('D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터')
            file_list = get_file_list(folder_path)
            while True:
                print("1: Ioniq5")
                print("2: Kona_EV")
                print("3: Porter_EV")
                car = int(input("Select Car you want to calculate: "))
                if car == 1:
                    EV = ioniq5
                elif car == 2:
                    EV = kona_EV
                elif car == 3:
                    EV = porter_EV
                else:
                    print("Invalid choice. Please try again.")
                    continue

                folder_path = os.path.join(folder_path, EV)
                process_files_energy(file_list, folder_path, EV)
                break
            break

        elif choice == 2:

            break

        elif choice == 3:

            break

        elif choice == 4:
            print("Quitting the program.")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
