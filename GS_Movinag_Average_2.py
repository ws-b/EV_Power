import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# 1. 주어진 디렉토리에서 파일 목록을 가져오기
directory_path = '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip'
file_lists = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# 2. 파일 목록을 그룹화하기
grouped_files = defaultdict(list)
for file in file_lists:
    key = file[:11]
    grouped_files[key].append(file)

for key, files in grouped_files.items():
    with PdfPages(os.path.join(directory_path, f'{key}_MA.pdf')) as pdf:
        for file in tqdm(files):
            filepath = os.path.join(directory_path, file)
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])  # Convert 'time' column to datetime format

            # Convert 'time' column to elapsed seconds from the start
            df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

            # 3. 데이터에서 Model_Energy와 BMS_Energy 계산하기
            df['Model_Energy'] = (df['Power'].cumsum() * 2) / 3600
            df['BMS_Energy'] = (df['Power_IV'].cumsum() * 2) / 3600

            # 4. Energy_Ratio와 그에 대한 이동 평균 계산하기
            df['Energy_Ratio'] = df['Model_Energy'] / df['BMS_Energy']
            df['Energy_Ratio_MA_2s'] = df['Energy_Ratio'].rolling(window=1).mean()
            df['Energy_Ratio_MA_1min'] = df['Energy_Ratio'].rolling(window=30).mean()
            df['Energy_Ratio_MA_5min'] = df['Energy_Ratio'].rolling(window=150).mean()

            # 5. Energy_Ratio와 그에 대한 이동 평균을 시각화하기
            plt.figure(figsize=(10, 7))
            plt.plot(df['time'], df['Energy_Ratio_MA_2s'], label='2s MA')
            plt.plot(df['time'], df['Energy_Ratio_MA_1min'], label='1min MA')
            plt.plot(df['time'], df['Energy_Ratio_MA_5min'], label='5min MA')
            plt.title(f'Energy Ratio and its Moving Averages for {file}')
            plt.xlabel('Elapsed Time (seconds)')
            plt.ylabel('Energy Ratio')
            plt.legend(loc='upper left')
            plt.ylim(1, 3)
            plt.tight_layout()

            # 6. PDF 파일에 그래프 추가하기
            pdf.savefig()
            plt.close()
