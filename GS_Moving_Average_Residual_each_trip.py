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
    key = file[:11]  # 파일 이름 기반으로 그룹화 (조정 가능)
    grouped_files[key].append(file)

for key, files in grouped_files.items():
    with PdfPages(os.path.join(directory_path, f'{key}_Residual_MA.pdf')) as pdf:  # Residual MA PDF
        for file in tqdm(files):
            filepath = os.path.join(directory_path, file)
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])  # 'time' 컬럼을 datetime 형식으로 변환

            # 'time' 컬럼을 시작부터 경과된 시간(초)으로 변환
            df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

            # 'Power'와 'Power_IV'의 차이(Residual) 계산
            df['Residual'] = df['Power'] - df['Power_IV']

            # Residual에 대한 이동 평균 계산
            df['Residual_MA_10s'] = df['Residual'].rolling(window=5).mean()
            df['Residual_MA_1min'] = df['Residual'].rolling(window=30).mean()
            df['Residual_MA_5min'] = df['Residual'].rolling(window=150).mean()

            # Residual과 그에 대한 이동 평균 시각화
            plt.figure(figsize=(10, 7))
            plt.plot(df['time'], df['Residual_MA_10s'], label='10s MA')
            plt.plot(df['time'], df['Residual_MA_1min'], label='1min MA')
            plt.plot(df['time'], df['Residual_MA_5min'], label='5min MA')
            plt.title(f'Residual and its Moving Averages for {file}')
            plt.xlabel('Elapsed Time (seconds)')
            plt.ylabel('Residual')
            plt.legend(loc='upper left')
            plt.tight_layout()

            # PDF 파일에 그래프 추가
            pdf.savefig()
            plt.close()