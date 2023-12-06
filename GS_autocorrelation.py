import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages

directory_path = '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/test_case'  # 여기에 실제 디렉토리 경로를 입력하세요.
file_lists = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# 절대값 평균으로 정규화하는 함수 정의
def normalize_by_abs_mean(series):
    return series / series.abs().mean()

# 파일 목록을 그룹화하기
grouped_files = defaultdict(list)
for file in file_lists:
    key = file[:11]  # 파일 이름에서 특정 부분을 기준으로 그룹화
    grouped_files[key].append(file)

lags = [10, 30, 50]  # 30초, 60초, 300초에 해당하는 레코드 수

# 각 그룹별로 처리
for key, files in grouped_files.items():
    with PdfPages(os.path.join(directory_path, f'{key}_autocorrelation.pdf')) as pdf:
        for file in tqdm(files):
            filepath = os.path.join(directory_path, file)
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])
            df['time_seconds'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
            df['Power_Diff'] = normalize_by_abs_mean(df['Power'] - df['Power_IV'])

            # 이동 평균 계산
            for lag in lags:
                df[f'MA_{lag}'] = df['Power_Diff'].rolling(window=lag).mean()

            # 자기상관도 계산
            autocorrelation_values = {lag: [] for lag in lags}
            for lag in lags:
                autocorr_values = [df['Power_Diff'].iloc[i:i + lag].autocorr() for i in range(len(df) - lag)]
                autocorrelation_values[lag] = autocorr_values

            # 이동 평균 및 자기상관도 시각화
            fig, axs = plt.subplots(3, 1, figsize=(15, 14))  # 2개의 서브플롯 생성

            # 이동 평균 시각화
            for lag in lags:
                axs[0].plot(df['time_seconds'], df[f'MA_{lag}'], label=f'MA {lag*2} seconds')
            axs[0].set_title(f'Moving Averages for {file}')
            axs[0].set_xlabel('Time (seconds)')
            axs[0].set_ylabel('Moving Average')
            axs[0].legend()

            # 자기상관도 시각화
            for lag, values in autocorrelation_values.items():
                axs[1].plot(df['time_seconds'][:len(values)], values, label=f'Lag {lag*2} seconds')
            axs[1].set_title(f'Autocorrelation for Different Lags Over Time for {file}')
            axs[1].set_xlabel('Time (seconds)')
            axs[1].set_ylabel('Autocorrelation')
            axs[1].legend()

            # Autocorrelation plot
            pd.plotting.autocorrelation_plot(df['Power_Diff'], ax=axs[2])
            axs[2].set_title(
                f'Autocorrelation of {("Residual")} for {file}')
            axs[2].set_xlabel('Lag')
            axs[2].set_ylabel('Autocorrelation')
            axs[2].set_xlim(0, 600)

            plt.tight_layout()
            pdf.savefig()  # PDF 파일에 그래프 추가
            plt.close()
