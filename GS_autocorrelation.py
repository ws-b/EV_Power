import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import os
from tqdm import tqdm
from collections import defaultdict

directory_path = '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip'
file_lists = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# 절대값 평균으로 정규화하는 함수 정의
def normalize_by_abs_mean(series):
    return series / series.abs().mean()

# 파일 목록을 그룹화하기
grouped_files = defaultdict(list)
for file in file_lists:
    key = file[:11]  # 파일 이름에서 특정 부분을 기준으로 그룹화
    grouped_files[key].append(file)

# 각 그룹별로 처리
for key, files in grouped_files.items():
    with PdfPages(os.path.join(directory_path, f'{key}_graphs.pdf')) as pdf:
        for file in tqdm(files):
            filepath = os.path.join(directory_path, file)
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])

            # 'time' 컬럼을 시작부터 경과된 시간(초)으로 변환
            df['time_seconds'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

            # 'Power'와 'Power_IV'의 차이 계산 및 정규화
            df['Power_Diff'] = normalize_by_abs_mean(df['Power'] - df['Power_IV'])

            # 이동 평균 계산
            df['Power_Diff_MA_10s'] = df['Power_Diff'].rolling(window=5).mean()
            df['Power_Diff_MA_1min'] = df['Power_Diff'].rolling(window=30).mean()
            df['Power_Diff_MA_5min'] = df['Power_Diff'].rolling(window=150).mean()

            # 모델 에너지와 BMS 에너지 계산
            df['Model_Energy'] = (df['Power'].cumsum() * 2) / 3600
            df['BMS_Energy'] = (df['Power_IV'].cumsum() * 2) / 3600

            # Power_Diff 및 이동 평균의 자기상관도를 계산하고 시각화하기
            for column, label in zip(['Power_Diff', 'Power_Diff_MA_10s', 'Power_Diff_MA_1min', 'Power_Diff_MA_5min'],
                                     ['Raw', '10 seconds', '1 minute', '5 minutes']):
                fig, axs = plt.subplots(1, 2, figsize=(20, 7))  # 1 row, 2 columns, each plot is 10x7

                # Autocorrelation plot
                pd.plotting.autocorrelation_plot(df[column].dropna(), ax=axs[0])
                axs[0].set_title(f'Autocorrelation of {("Power Difference" if label == "Raw" else "Power Difference Moving Average")} ({label}) for {file}')
                axs[0].set_xlabel('Lag')
                axs[0].set_ylabel('Autocorrelation')

                # Energy plot
                axs[1].plot(df['time_seconds'], df['Model_Energy'], label='Model Energy')
                axs[1].plot(df['time_seconds'], df['BMS_Energy'], label='BMS Energy')
                axs[1].set_title(f'Model and BMS Energy over Time for {file}')
                axs[1].set_xlabel('Elapsed Time (seconds)')
                axs[1].set_ylabel('Energy (kWh)')
                axs[1].legend()

                plt.tight_layout()
                pdf.savefig()  # PDF 파일에 그래프 추가
                plt.close()
