import os
import glob
import pandas as pd
from tqdm import tqdm

directory = r'D:\SamsungSTF\Data\KOTI\2019년_팅크웨어_포인트경로데이터'
processed_path = r'D:\SamsungSTF\Processed_Data\KOTI'

file_lists = glob.glob(os.path.join(directory, "*.csv"))

for file in tqdm(file_lists):
    df = pd.read_csv(file)

    if 'obu_id' in df.columns and 'time' in df.columns:
        obu_ids = df['obu_id'].unique()

        for obu_id in obu_ids:
            obu_df = df[df['obu_id'] == obu_id]

            filtered_df = obu_df[['time', 'x', 'y']].copy()

            try:
                filtered_df['time'] = pd.to_datetime(filtered_df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            except ValueError as e:
                print(f"Date format error in file '{file}' for OBU ID: {obu_id}: {e}")
                continue

            if filtered_df['time'].isna().all():
                print(f"파일 '{file}'에서 유효한 날짜 형식을 찾지 못했습니다. OBU ID: {obu_id}")
                continue

            date_str = filtered_df['time'].dt.strftime('%Y%m%d').iloc[0]

            save_path = os.path.join(processed_path, f"{date_str}_{obu_id}.csv")

            # 파일이 이미 존재하면 스킵
            if os.path.exists(save_path):
                print(f"파일이 이미 존재합니다: {save_path}. 스킵합니다.")
                continue

            try:
                filtered_df.to_csv(save_path, index=False)
                print(f"파일 저장됨: {save_path}")
            except Exception as e:
                print(f"파일 저장 중 에러 발생: {e}")
    else:
        print(f"'obu_id' 또는 'time' 열이 없음: {file}")
