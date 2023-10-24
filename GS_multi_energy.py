import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# 주어진 디렉토리에서 파일 목록을 가져오기
directory_path = '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip'
file_lists = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# 파일 목록을 그룹화하기
grouped_files = defaultdict(list)
for file in file_lists:
    key = file[:11]  # 파일 이름에서 고유한 키 추출
    grouped_files[key].append(file)

# 그룹별로 파일 처리
for key, files in grouped_files.items():
    # 파일이 9개 미만인 경우 건너뜀
    if len(files) < 9:
        print(f"Group {key} has less than 9 files, skipping...")
        continue

    # 3x3 그리드로 9개의 서브플롯 생성
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))  # 크기는 조절 가능
    axs = axs.flatten()  # 2D 배열을 1D로 변환하여 인덱싱 용이하게 함

    # 서브플롯에 그래프 그리기
    for i, file in enumerate(tqdm(files[:9])):  # 처음 9개 파일만 처리
        filepath = os.path.join(directory_path, file)
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
        df['Model_Energy'] = (df['Power'].cumsum() * 2) / 3600
        df['BMS_Energy'] = (df['Power_IV'].cumsum() * 2) / 3600

        axs[i].plot(df['time'], df['Model_Energy'], 'b', label='Model Energy')
        axs[i].plot(df['time'], df['BMS_Energy'], 'r', label='BMS Energy')
        axs[i].set_title(f'File: {file}')
        axs[i].set_xlabel('Elapsed Time (seconds)')
        axs[i].set_ylabel('Energy (kWh)')
        axs[i].legend(loc='upper left')

    plt.tight_layout()
    plt.show()  # 화면에 그래프 표시
