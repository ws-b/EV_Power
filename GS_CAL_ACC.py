import os
import numpy as np
import pandas as pd

mac_folder_path = '/Users/woojin/Downloads/한국에너지공과대학교_샘플데이터/processed/'
mac_save_path = '/Users/woojin/Downloads/한국에너지공과대학교_샘플데이터/speed-acc/'
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 Processed\\'
win_save_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'

folder_path = mac_folder_path
save_path = mac_save_path
# get a list of all files in the folder with the .csv extension
# file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
# for file_list in file_lists:
#     file = open(folder_path + file_list, "r")
    file = '/Users/woojin/Downloads/한국에너지공과대학교_샘플데이터/processed/bms.01241228177.02_parsed.csv'
    # Load CSV file into a pandas DataFrame
    data = pd.read_csv(file)

    print(data.columns)
    # km/h 단위의 속도를 m/s 단위로 변환
    emobility_spd_m_s = data['emobility_spd'] * 0.27778

    # Calculate the acceleration
    acceleration = np.gradient(emobility_spd_m_s, time)

    # 마지막 가속도 값 가져오기
    last_acceleration = acceleration[-1]

    # 마지막 가속도 값을 가속도 배열에 두 번 추가하기
    acceleration = np.append(acceleration, [last_acceleration, last_acceleration])

    # 각 컬럼을 하나의 DataFrame으로 합치기
    data_save = pd.DataFrame({
        'Time': time,
        'Speed': emobility_spd_m_s,
        'Acceleration': acceleration,
        'trip_chrg_pw': trip_chrg_pw,
        'trip_dischrg_pw': trip_dischrg_pw,
        'soc': soc,
        'soh': soh
    })

    # csv 파일로 저장하기
    data_save.to_csv(f"{save_path}{str(device_no.iloc[0])}.csv", index=False)