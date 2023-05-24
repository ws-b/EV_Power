import os
import numpy as np
import pandas as pd
import csv
import datetime


# 파일이 들어있는 폴더 경로
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\speed-acc\\'
mac_folder_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터/'
win_save_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\trip_by_trip\\'
mac_save_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터 Processed/'

folder_path = win_folder_path
save_path = win_save_path

def get_file_list(folder_path):
    # 폴더 내의 모든 파일 리스트 가져오기
    file_list = os.listdir(folder_path)
    txt_files = []
    for file in file_list:
        if file.endswith('.txt'):
            txt_files.append(file)
    return txt_files

# 파일 리스트 가져오기
files = get_file_list(folder_path)
files.sort()

#for file in files:
    # csv 파일 불러오기
#    df = pd.read_csv(folder_path + file, sep=',')
