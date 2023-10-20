import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import os

# 주어진 디렉토리에서 파일 목록을 가져오기
directory_path = '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip'  # 사용자의 디렉토리 경로를 입력하세요
file_lists = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# 파일 목록을 그룹화하기
grouped_files = defaultdict(list)
for file in file_lists:
    key = file[:11]
    grouped_files[key].append(file)

# 각 그룹별로 처리
for key, files in grouped_files.items():
    dataframes = []
    differences = []  # Power와 Power_IV의 차이를 저장할 리스트
    differences_MA_10s = []  # 10초 이동 평균
    differences_MA_1min = []  # 1분 이동 평균
    differences_MA_5min = []  # 5분 이동 평균

    # 각 그룹별로 파일을 병합하기
    for file in files:
        filepath = os.path.join(directory_path, file)
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        dataframes.append(df)

    merged_data = pd.concat(dataframes, ignore_index=True)
    merged_data = merged_data.sort_values(by='time').reset_index(drop=True)

    # 병합된 데이터에서 Power와 Power_IV의 차이 계산하기
    merged_data['Power_Difference'] = merged_data['Power'] - merged_data['Power_IV']

    # Power_Difference와 그에 대한 이동 평균 계산하기
    merged_data['Power_Diff_MA_10s'] = merged_data['Power_Difference'].rolling(window=5).mean()
    merged_data['Power_Diff_MA_1min'] = merged_data['Power_Difference'].rolling(window=30).mean()
    merged_data['Power_Diff_MA_5min'] = merged_data['Power_Difference'].rolling(window=150).mean()

    differences.extend(merged_data['Power_Difference'].tolist())
    differences_MA_10s.extend(merged_data['Power_Diff_MA_10s'].tolist())
    differences_MA_1min.extend(merged_data['Power_Diff_MA_1min'].tolist())
    differences_MA_5min.extend(merged_data['Power_Diff_MA_5min'].tolist())

    # 각 그룹별 Power_Difference의 분포와 이동 평균 분포를 시각화하기
    plt.figure(figsize=(10, 7))
    sns.kdeplot(differences, label='Power Difference')
    sns.kdeplot(differences_MA_10s, label='10s MA')
    sns.kdeplot(differences_MA_1min, label='1min MA')
    sns.kdeplot(differences_MA_5min, label='5min MA')
    plt.title(f'Distribution of Power Differences and its Moving Averages for {key}')
    plt.xlabel('Power Difference (Power - Power_IV)')
    plt.ylabel('Density')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.show()