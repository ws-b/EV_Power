import os
import pandas as pd

# folder path where the files are stored
win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\한국에너지공과대학교_샘플데이터\ioniq5'

folder_path = os.path.normpath(win_folder_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()