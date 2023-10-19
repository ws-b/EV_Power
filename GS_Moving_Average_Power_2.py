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

            # Calculate the ratio of Power to Power_IV
            df['Power_Ratio'] = df['Power'] / df['Power_IV']

            # Calculate the moving averages for the Power_Ratio
            df['Power_Ratio_MA_10s'] = df['Power_Ratio'].rolling(window=5).mean()
            df['Power_Ratio_MA_1min'] = df['Power_Ratio'].rolling(window=30).mean()
            df['Power_Ratio_MA_5min'] = df['Power_Ratio'].rolling(window=150).mean()

            # Visualize the Power_Ratio and its moving averages
            plt.figure(figsize=(10, 7))
            plt.plot(df['time'], df['Power_Ratio_MA_10s'], label='10s MA')
            plt.plot(df['time'], df['Power_Ratio_MA_1min'], label='1min MA')
            plt.plot(df['time'], df['Power_Ratio_MA_5min'], label='5min MA')
            plt.title(f'Power Ratio and its Moving Averages for {file}')
            plt.xlabel('Elapsed Time (seconds)')
            plt.ylabel('Power Ratio')
            plt.legend(loc='upper left')
            plt.tight_layout()

            # Save the graph to the PDF file
            pdf.savefig()
            plt.close()