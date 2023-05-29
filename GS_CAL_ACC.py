import os
import numpy as np
import pandas as pd

mac_folder_path = '/Users/woojin/Downloads/한국에너지공과대학교_샘플데이터/processed/'
mac_save_path = '/Users/woojin/Downloads/한국에너지공과대학교_샘플데이터/speed-acc/'
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\processed\\'
win_save_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\speed-acc\\'

folder_path = win_folder_path
save_path = win_save_path
# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
for file_list in file_lists:
    file = open(folder_path + file_list, "r")

    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(file, dtype={'device_no': str, 'measured_month': str})

    # index 기준으로 거꾸로 정렬
    df = df[::-1]

    # 속도 단위를 km/h에서 m/s로 변환
    df['emobility_spd_m_per_s'] = df['emobility_spd'] * 0.27778

    # 시간과 변환된 속도의 변화를 계산합니다.
    df['time'] = df['time'].str.strip()
    df['time'] = pd.to_datetime(df['time'], format='%y-%m-%d %H:%M:%S')
    df['time_diff'] = df['time'].diff().dt.total_seconds()
    df['emobility_spd_m_per_s'] = df['emobility_spd'] * 0.27778
    df['spd_diff'] = df['emobility_spd_m_per_s'].diff()

    # 가속도 계산
    df['acceleration'] = df['spd_diff'] / df['time_diff']

    # NaN 값을 0으로 바꿔주거나, 원하는 값으로 채워줍니다.
    df['acceleration'] = df['acceleration'].replace(np.nan, 0)    # 각 컬럼을 하나의 DataFrame으로 합치기

    data_save = df[['time', 'emobility_spd_m_per_s', 'acceleration', 'trip_chrg_pw', 'trip_dischrg_pw', 'chrg_cable_conn', 'soc', 'soh']].copy()

    # csv 파일로 저장하기
    data_save.to_csv(f"{save_path}{df['device_no'].iloc[0].replace(' ', '')}{'-0' + df['measured_month'].iloc[0][-2:].replace(' ', '')}.csv", index=False)
