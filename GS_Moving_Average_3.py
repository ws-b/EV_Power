import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


def classify_data(df):
    df['Energy_Ratio_CV'] = df['Energy_Ratio'].rolling(window=150).std() / df['Energy_Ratio'].rolling(window=150).mean()
    threshold = 0.1
    if df['Energy_Ratio_CV'].mean() > threshold:
        return 'High Variability'
    else:
        return 'Low Variability'


# 1. 주어진 디렉토리에서 파일 목록을 가져오기
directory_path = '/Users/wsong/Documents/삼성미래과제/한국에너지공과대학교_샘플데이터/trip_by_trip'
file_lists = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# 2. 파일 목록을 그룹화하고 정렬하기
grouped_files = defaultdict(list)
for file in file_lists:
    key = file[:11]
    grouped_files[key].append(file)

# 각 그룹 내의 파일 목록 정렬
for key in grouped_files:
    grouped_files[key].sort()

for key, files in grouped_files.items():
    high_variability_path = os.path.join(directory_path, f'{key}_High_Variability_MA.pdf')
    low_variability_path = os.path.join(directory_path, f'{key}_Low_Variability_MA.pdf')

    high_variability_pdf = PdfPages(high_variability_path)
    low_variability_pdf = PdfPages(low_variability_path)

    for file in tqdm(files):
        filepath = os.path.join(directory_path, file)
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

        df['Model_Energy'] = (df['Power'].cumsum() * 2) / 3600
        df['BMS_Energy'] = (df['Power_IV'].cumsum() * 2) / 3600

        df['Energy_Ratio'] = df['Model_Energy'] / df['BMS_Energy']
        df['Energy_Ratio_MA_2s'] = df['Energy_Ratio'].rolling(window=1).mean()
        df['Energy_Ratio_MA_1min'] = df['Energy_Ratio'].rolling(window=30).mean()
        df['Energy_Ratio_MA_5min'] = df['Energy_Ratio'].rolling(window=150).mean()

        # 분류 결정
        classification = classify_data(df)

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

        # 분류에 따라서 PDF에 저장
        if classification == 'High Variability':
            high_variability_pdf.savefig()
        else:
            low_variability_pdf.savefig()
        plt.close()

    # PDF 파일 닫기
    high_variability_pdf.close()
    low_variability_pdf.close()
