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
    key = file[:11]  # 파일 이름에서 특정 부분을 기준으로 그룹화 (여기서는 파일명의 처음 11자리)
    grouped_files[key].append(file)

# 각 그룹별로 처리
for key, files in grouped_files.items():
    dataframes = []

    # 각 그룹별로 파일을 병합하기
    for file in files:
        filepath = os.path.join(directory_path, file)
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])  # 'time' 열을 datetime 객체로 변환
        dataframes.append(df)

    merged_data = pd.concat(dataframes, ignore_index=True)
    merged_data = merged_data.sort_values(by='time').reset_index(drop=True)  # 시간 순으로 데이터 정렬

    # 병합된 데이터에서 Power와 Power_IV의 차이 계산하기
    merged_data['Power_Difference'] = merged_data['Power'] - merged_data['Power_IV']

    # Residual 값을 정규화하기 위해 절대값의 평균으로 나누기
    absolute_mean = merged_data['Power_Difference'].abs().mean()  # 절대값의 평균 계산
    merged_data['Normalized_Power_Difference'] = merged_data['Power_Difference'] / absolute_mean  # 정규화

    # Power_Difference와 그에 대한 이동 평균 계산하기
    merged_data['Power_Diff_MA_10s'] = merged_data['Normalized_Power_Difference'].rolling(window=5).mean()
    merged_data['Power_Diff_MA_1min'] = merged_data['Normalized_Power_Difference'].rolling(window=30).mean()
    merged_data['Power_Diff_MA_5min'] = merged_data['Normalized_Power_Difference'].rolling(window=150).mean()

    # 각 그룹별 Power_Difference의 분포와 이동 평균 분포를 시각화하기
    plt.figure(figsize=(10, 7))
    sns.kdeplot(merged_data['Normalized_Power_Difference'], label='Normalized Power Difference (2s)')
    sns.kdeplot(merged_data['Power_Diff_MA_10s'], label='10s MA')
    sns.kdeplot(merged_data['Power_Diff_MA_1min'], label='1min MA')
    sns.kdeplot(merged_data['Power_Diff_MA_5min'], label='5min MA')
    plt.title(f'Distribution of Normalized Power Differences and its Moving Averages for {key}')
    plt.xlabel('Normalized Power Difference (Power - Power_IV)')
    plt.ylabel('Density')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.xlim(0,1)
    plt.legend()
    plt.show()
