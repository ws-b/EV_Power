import pandas as pd
import numpy as np
import os
import shutil
from pyproj import Transformer
from tqdm import tqdm

# 경로 설정
processed_path = r'D:\SamsungSTF\Processed_Data\KOTI'
invalid_year_path = os.path.join(processed_path, 'invalid_year')

# 필요한 폴더가 없으면 생성
os.makedirs(invalid_year_path, exist_ok=True)

# 파일 로드
csv_files = [f for f in os.listdir(processed_path) if f.endswith('.csv')]

# 모든 CSV 파일 처리
for file in tqdm(csv_files):
    file_path = os.path.join(processed_path, file)
    df = pd.read_csv(file_path)

    # time 열을 datetime으로 변환
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # 잘못된 연도 찾기 (2018, 2019, 2020년 외의 값 확인)
    invalid_years = df[~df['time'].dt.year.isin([2018, 2019, 2020])]

    if not invalid_years.empty:
        # 파일을 삭제
        os.remove(file_path)
        continue

print("Processing and file moving completed.")
