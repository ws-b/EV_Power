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
    grouped_files[key].sort()  # 파일 이름을 정렬하여 순서를 일관되게 함
    
for key, files in grouped_files.items():
    # 처음 3개 파일을 위한 3x2 그리드 생성
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    for i, file in enumerate(files[:3]):  # 처음 3개 파일만 처리
        filepath = os.path.join(directory_path, file)
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

        # Left subplot: Power only
        axs[i, 0].plot(df['time'], df['Power'], 'b', label='Power')
        axs[i, 0].set_title(f'Power for {file}')
        axs[i, 0].set_xlabel('Elapsed Time (seconds)')
        axs[i, 0].set_ylabel('Power (kW)')
        axs[i, 0].legend(loc='upper left')

        # Right subplot: Difference between Power and Power2
        axs[i, 1].plot(df['time'], df['Power'] - df['Power2'], 'r', label='Difference (Power - Power2)')
        axs[i, 1].set_title(f'Power Difference for {file}')
        axs[i, 1].set_xlabel('Elapsed Time (seconds)')
        axs[i, 1].set_ylabel('Power Difference (kW)')
        axs[i, 1].legend(loc='upper left')

    plt.tight_layout()
    plt.show()  # 화면에 그래프 표시

    with PdfPages(os.path.join(directory_path, f'{key}_Power_cd.pdf')) as pdf:
        # 모든 파일들에 대해 1x2 그리드로 그래프를 그린 후 PDF에 저장
        for file in tqdm(files):
            filepath = os.path.join(directory_path, file)
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])
            df['time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

            fig, axs = plt.subplots(1, 2, figsize=(15, 7))

            # Left subplot: Power only
            axs[0].plot(df['time'], df['Power'], 'b', label='Power')
            axs[0].set_title(f'Power for {file}')
            axs[0].set_xlabel('Elapsed Time (seconds)')
            axs[0].set_ylabel('Power (kW)')
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
