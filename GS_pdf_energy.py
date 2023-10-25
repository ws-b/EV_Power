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
    with PdfPages(os.path.join(directory_path, f'{key}_Energy_cd.pdf')) as pdf:  # Energy PDF
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

            # 모델 에너지와 BMS 에너지를 동일한 그래프에 표시
            plt.figure(figsize=(10, 7))
            plt.plot(df['time'], df['Model_Energy'], 'b', label='Model Energy')  # 파란색으로 모델 에너지 표시
            plt.plot(df['time'], df['BMS_Energy'], 'r', label='BMS Energy')  # 빨간색으로 BMS 에너지 표시
            plt.title(f'Energy Comparison for {file}')
            plt.xlabel('Elapsed Time (seconds)')
            plt.ylabel('Energy (kWh)')
            plt.legend(loc='upper left')
            plt.tight_layout()

            # PDF 파일에 그래프 추가
            pdf.savefig()
            plt.close()
