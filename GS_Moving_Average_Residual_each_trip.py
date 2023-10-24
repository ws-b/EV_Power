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
    with PdfPages(os.path.join(directory_path, f'{key}_Residual_MA.pdf')) as pdf:  # Residual MA PDF
        for file in tqdm(files):
            filepath = os.path.join(directory_path, file)
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])

            # 'time' 컬럼을 시작부터 경과된 시간(초)으로 변환
            df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

            # 'Power'와 'Power_IV'의 차이(Residual) 계산
            df['Residual'] = df['Power'] - df['Power_IV']

            # 모델 에너지와 BMS 에너지 계산
            df['Model_Energy'] = (df['Power'].cumsum() * 2) / 3600
            df['BMS_Energy'] = (df['Power_IV'].cumsum() * 2) / 3600

            # Residual에 대한 이동 평균 계산
            df['Residual_MA_10s'] = df['Residual'].rolling(window=5).mean()
            df['Residual_MA_1min'] = df['Residual'].rolling(window=30).mean()
            df['Residual_MA_5min'] = df['Residual'].rolling(window=150).mean()

            # 두 개의 서브플롯 생성
            fig, axs = plt.subplots(1, 2, figsize=(20, 7))  # 1 row, 2 columns, and figure size of 20x7 inches

            # 첫 번째 서브플롯 (왼쪽): Residual과 그에 대한 이동 평균
            axs[0].plot(df['time'], df['Residual_MA_10s'], label='10s MA')
            axs[0].plot(df['time'], df['Residual_MA_1min'], label='1min MA')
            axs[0].plot(df['time'], df['Residual_MA_5min'], label='5min MA')
            axs[0].set_title(f'Residual and its Moving Averages for {file}')
            axs[0].set_xlabel('Elapsed Time (seconds)')
            axs[0].set_ylabel('Residual')
            axs[0].legend(loc='upper left')

            # 두 번째 서브플롯 (오른쪽): 모델 에너지와 BMS 에너지
            axs[1].plot(df['time'], df['Model_Energy'], label='Model Energy')
            axs[1].plot(df['time'], df['BMS_Energy'], label='BMS Energy')
            axs[1].set_title(f'Model and BMS Energy for {file}')
            axs[1].set_xlabel('Elapsed Time (seconds)')
            axs[1].set_ylabel('Energy (kWh)')
            axs[1].legend(loc='upper left')

            plt.tight_layout()

            # PDF 파일에 그래프 추가
            pdf.savefig()
            plt.close()
