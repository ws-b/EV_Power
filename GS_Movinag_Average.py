import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 1. 주어진 디렉토리에서 파일 목록을 가져오기
directory_path = '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip'  # 사용자의 디렉토리 경로를 입력하세요
file_lists = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# 2. 파일 목록을 그룹화하기
grouped_files = defaultdict(list)
for file in file_lists:
    key = file[:11]
    grouped_files[key].append(file)

# 각 그룹별로 처리
for key, files in grouped_files.items():
    dataframes = []
    ratios = []
    ratios_MA_2s = []
    ratios_MA_1min = []
    ratios_MA_5min = []

    # 3. 각 그룹별로 파일을 병합하기
    for file in files:
        filepath = os.path.join(directory_path, file)
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        dataframes.append(df)

    merged_data = pd.concat(dataframes, ignore_index=True)
    merged_data = merged_data.sort_values(by='time').reset_index(drop=True)

    # 4. 병합된 데이터에서 Model_Energy와 BMS_Energy 계산하기
    merged_data['Model_Energy'] = (merged_data['Power'].cumsum() * 2) / 3600
    merged_data['BMS_Energy'] = (merged_data['Power_IV'].cumsum() * 2) / 3600

    # 5. Energy_Ratio와 그에 대한 이동 평균 계산하기
    merged_data['Energy_Ratio'] = merged_data['Model_Energy'] / merged_data['BMS_Energy']
    merged_data['Energy_Ratio_MA_2s'] = merged_data['Energy_Ratio'].rolling(window=1).mean()
    merged_data['Energy_Ratio_MA_1min'] = merged_data['Energy_Ratio'].rolling(window=30).mean()
    merged_data['Energy_Ratio_MA_5min'] = merged_data['Energy_Ratio'].rolling(window=150).mean()

    ratios.extend(merged_data['Energy_Ratio'].tolist())
    ratios_MA_2s.extend(merged_data['Energy_Ratio_MA_2s'].tolist())
    ratios_MA_1min.extend(merged_data['Energy_Ratio_MA_1min'].tolist())
    ratios_MA_5min.extend(merged_data['Energy_Ratio_MA_5min'].tolist())

    # 6. 각 그룹별 Energy_Ratio의 분포와 이동 평균 분포를 시각화하기
    plt.figure(figsize=(10, 7))
    sns.kdeplot(ratios, label='Energy_Ratio')
    sns.kdeplot(ratios_MA_2s, label='2s MA')
    sns.kdeplot(ratios_MA_1min, label='1min MA')
    sns.kdeplot(ratios_MA_5min, label='5min MA')
    plt.title(f'Distribution of Energy Ratios and its Moving Averages for {key}')
    plt.xlabel('Energy Ratio (Model_Energy/BMS_Energy)')
    plt.ylabel('Density')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.xlim(0.9, 1.6)
    plt.legend()
    plt.show()
