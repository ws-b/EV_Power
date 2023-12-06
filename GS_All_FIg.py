import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
from tqdm import tqdm

# 주어진 디렉토리에서 파일 목록을 가져오기
directory_path = '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/test_case'
file_lists = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# 파일 목록을 그룹화하기
grouped_files = defaultdict(list)
for file in file_lists:
    key = file[:11]  # 파일 이름에서 특정 부분을 기준으로 그룹화
    grouped_files[key].append(file)

lags = [1, 8, 15, 30]  # 30초, 60초, 300초에 해당하는 레코드 수

# 각 그룹별로 처리
for key, files in grouped_files.items():
    with PdfPages(os.path.join(directory_path, f'{key}_comprehensive_analysis.pdf')) as pdf:
        for file in tqdm(files):
            filepath = os.path.join(directory_path, file)
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])
            df['time_seconds'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

            # Power Difference 및 이동 평균 계산
            df['Residual'] = df['Power2'] - df['Power_IV']

            # 이동 평균 계산
            for lag in lags:
                df[f'Residual_MA_{lag}'] = df['Residual'].rolling(window=lag).mean()

            # 에너지 계산
            df['Model_Energy'] = (df['Power2'].cumsum() * 2) / 3600
            df['BMS_Energy'] = (df['Power_IV'].cumsum() * 2) / 3600

            # 그래프 그리기
            fig, axs = plt.subplots(4, 1, figsize=(15, 15))

            autocorrelation_values = {lag: [] for lag in lags}
            for lag in lags:
                autocorr_values = [df['Residual'].iloc[i:i + lag].autocorr() for i in range(len(df) - lag)]
                autocorrelation_values[lag] = autocorr_values
            # 첫 번째 행: Residual Moving Average 시각화
            for lag in lags:
                axs[0].plot(df['time_seconds'], df[f'Residual_MA_{lag}'], label=f'MA {lag*2} seconds')
            axs[0].set_title(f'Moving Averages for {file}')
            axs[0].set_xlabel('Time (seconds)')
            axs[0].set_ylabel('Moving Average')
            axs[0].legend(loc='upper left')

            # 두 번째 행: 자기상관도 시각화
            for lag, values in autocorrelation_values.items():
                axs[1].plot(df['time_seconds'][:len(values)], values, label=f'Lag {lag*2} seconds')
            axs[1].set_title(f'Autocorrelation for Different Lags Over Time for {file}')
            axs[1].set_xlabel('Time (seconds)')
            axs[1].set_ylabel('Autocorrelation')
            axs[1].legend(loc='upper left')
            axs[1].grid(True)

            # 세 번째 행: 파워 그래프
            axs[2].plot(df['time'], df['Power2'], 'b', label='Model Energy')  # 파란색으로 모델 에너지 표시
            axs[2].plot(df['time'], df['Power_IV'], 'r', label='BMS Energy')  # 빨간색으로 BMS 에너지 표시
            axs[2].set_title(f'Power Comparison for {file}')
            axs[2].set_xlabel('Elapsed Time (seconds)')
            axs[2].set_ylabel('Power (W)')
            axs[2].legend(loc='upper left')

            # 네 번째 행: 에너지 그래프
            axs[3].plot(df['time'], df['Model_Energy'], 'b', label='Model Energy')  # 파란색으로 모델 에너지 표시
            axs[3].plot(df['time'], df['BMS_Energy'], 'r', label='BMS Energy')  # 빨간색으로 BMS 에너지 표시
            axs[3].set_title(f'Energy Comparison for {file}')
            axs[3].set_xlabel('Elapsed Time (seconds)')
            axs[3].set_ylabel('Energy (kWh)')
            axs[3].legend(loc='upper left')

            plt.tight_layout()
            pdf.savefig()
            plt.close()
