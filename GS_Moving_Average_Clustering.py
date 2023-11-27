import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm

# 1. 주어진 디렉토리에서 파일 목록을 가져오기
directory_path = '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip'
file_lists = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# 2. 파일 목록을 그룹화하기
grouped_files = defaultdict(list)
for file in file_lists:
    key = file[:11]
    grouped_files[key].append(file)

# 각 그룹별로 클러스터링 실행 및 결과 시각화
for key, files in tqdm(grouped_files.items()):
    original_dfs = []  # 원본 데이터를 저장할 리스트
    ratio_data = []  # Energy Ratio 데이터를 저장할 리스트

    for file in files:
        filepath = os.path.join(directory_path, file)
        df = pd.read_csv(filepath)

        df['time'] = pd.to_datetime(df['time'])
        df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

        df['Model_Energy'] = (df['Power'].cumsum() * 2) / 3600
        df['BMS_Energy'] = (df['Power_IV'].cumsum() * 2) / 3600
        df['Energy_Ratio'] = df['Model_Energy'] / df['BMS_Energy']
        df['Energy_Ratio_MA_1min'] = df['Energy_Ratio'].rolling(window=30).mean()
        df['Energy_Ratio_STD'] = df['Energy_Ratio'].rolling(window=30).std()

        original_dfs.append(df)
        ratio_data.append(df[['Energy_Ratio_MA_1min', 'Energy_Ratio_STD']].dropna())

        # 모든 Energy Ratio 데이터를 하나로 통합
    ratio_df = pd.concat(ratio_data)

    # 클러스터링을 위한 데이터 준비
    X = ratio_df.values

    # K-Means 클러스터링 실행
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    labels = kmeans.labels_

    # 클러스터링된 각 Trip의 Energy 그래프 생성
    for cluster_num in range(3):  # 클러스터의 수에 따라 조정
        plt.figure(figsize=(15, 10))

        for i, df in enumerate(original_dfs):
            if labels[i] == cluster_num:  # 클러스터 레이블에 따라 Trip 선택
                plt.plot(df['time'], df['Model_Energy'], label=f'Model Energy Trip {i}')
                plt.plot(df['time'], df['BMS_Energy'], label=f'BMS Energy Trip {i}', linestyle='--')

        plt.title(f'Energy Over Time for Trips in Cluster {cluster_num} of Group {key}')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.legend()
        plt.show()

    # 클러스터링 결과 시각화
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title(f'Clustering Result for Group {key}')
    plt.xlabel('1min Moving Average of Energy Ratio')
    plt.ylabel('Standard Deviation of Energy Ratio')
    plt.show()