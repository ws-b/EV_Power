import os
import pandas as pd
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
    grouped_files[key].sort()  # 파일 이름을 정렬하여 순서를 일관되게 함

for key, files in grouped_files.items():
    # 모든 파일들에 대해 1x2 그리드로 그래프를 그린 후 PDF에 저장
    with PdfPages(os.path.join(directory_path, f'{key}_Power_Energy_cd.pdf')) as pdf:
        for file in tqdm(files):
            filepath = os.path.join(directory_path, file)
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])
            df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

            # Energy 계산 (간단한 방법으로 시간 간격을 고정 가정)
            df['Energy'] = df['Power'].cumsum() * (df['time'][1] - df['time'][0]) / 3600  # kWh 단위로 변환

            fig, axs = plt.subplots(1, 2, figsize=(15, 7))

            # Left subplot: Energy over Time
            axs[0].plot(df['time'], df['Energy'], 'b', label='Energy')
            axs[0].set_title(f'Energy over Time for {file}')
            axs[0].set_xlabel('Elapsed Time (seconds)')
            axs[0].set_ylabel('Energy (kWh)')
            axs[0].legend(loc='upper left')

            # Right subplot: Difference between Power and Power2
            axs[1].plot(df['time'], df['Power'] - df['Power2'], 'r', label='Difference (Power - Power2)')
            axs[1].set_title(f'Power Difference for {file}')
            axs[1].set_xlabel('Elapsed Time (seconds)')
            axs[1].set_ylabel('Power Difference (kW)')
            axs[1].legend(loc='upper left')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
