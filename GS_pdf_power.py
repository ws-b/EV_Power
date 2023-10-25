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
    with PdfPages(os.path.join(directory_path, f'{key}_Power_cd.pdf')) as pdf:  # Power PDF
        for file in tqdm(files):
            filepath = os.path.join(directory_path, file)
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])

            # 'time' 컬럼을 시작부터 경과된 시간(초)으로 변환
            df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

            # Power와 Power_IV를 동일한 그래프에 표시
            plt.figure(figsize=(10, 7))
            plt.plot(df['time'], df['Power'], 'b', label='Power')
            plt.plot(df['time'], df['Power_IV'], 'r', label='Power_IV')
            plt.title(f'Power Comparison for {file}')
            plt.xlabel('Elapsed Time (seconds)')
            plt.ylabel('Power (kW)')
            plt.legend(loc='upper left')
            plt.tight_layout()

            # PDF 파일에 그래프 추가
            pdf.savefig()
            plt.close()
