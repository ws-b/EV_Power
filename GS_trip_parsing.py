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
    file = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\speed-acc\\01241228177-02.csv'
    df = pd.read_csv(file, sep=',')
    df['time'] = pd.to_datetime(df['time'])

    # pandas DataFrame을 numpy 배열로 변환
    data = df.values

    # DATETIME을 초로 변환
    time_in_seconds = (data[:, 0] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

    # Speed가 0인 기간이 300초 이상인 경우, 혹은 chrg_cable_connect가 1인 경우를 찾기
    split_conditions = np.logical_or((data[:, 1] == 0) & (np.diff(time_in_seconds, prepend=time_in_seconds[0]) >= 300), data[:, 5] == 1)

    # 위 조건에 해당하는 지점에서 데이터 분할
    split_indices = np.where(split_conditions)[0]
    split_indices = np.append(split_indices, len(data))  # 마지막 인덱스 추가
"""
    # 각각의 데이터 조각을 별도의 csv 파일로 저장
    for i in range(len(split_indices) - 1):
        split_data = data[split_indices[i]:split_indices[i + 1]]
        pd.DataFrame(split_data, columns=df.columns).to_csv(f'output_{i}.csv', index=False)
        
"""