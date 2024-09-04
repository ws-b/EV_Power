import os
import glob
import pandas as pd
from tqdm import tqdm
# CSV 파일이 있는 경로 지정
directory = r'D:\SamsungSTF\Data\KOTI\2019년_팅크웨어_포인트경로데이터'
processed_path = r'D:\SamsungSTF\Processed_Data\KOTI'

# 모든 CSV 파일 경로를 리스트로 불러오기
csv_files = glob.glob(os.path.join(directory, "*.csv"))

for file in tqdm(csv_files):
    df = pd.read_csv(file)

    if 'obu_id' in df.columns and 'time' in df.columns:
        # 'obu_id' 열의 고유 값을 가져옴
        obu_ids = df['obu_id'].unique()

        for obu_id in obu_ids:
            # 같은 'obu_id'를 가진 데이터 필터링
            obu_df = df[df['obu_id'] == obu_id]

            # 필요한 열만 선택 (time, x, y, spd, acc)
            filtered_df = obu_df[['time', 'x', 'y', 'spd', 'acc']].copy()  # 명시적 복사

            # 'time' 열을 날짜 형식으로 변환
            try:
                filtered_df['time'] = pd.to_datetime(filtered_df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            except ValueError as e:
                print(f"Date format error in file '{file}' for OBU ID: {obu_id}: {e}")
                continue

            # 변환된 'time' 열이 datetime 형식인지 확인
            if filtered_df['time'].isna().all():
                print(f"파일 '{file}'에서 유효한 날짜 형식을 찾지 못했습니다. OBU ID: {obu_id}")
                continue  # 유효한 날짜 형식을 찾지 못한 경우 해당 파일은 처리하지 않고 넘어감

            # 첫 번째 값을 기준으로 '연-월-일' 추출
            date_str = filtered_df['time'].dt.strftime('%Y%m%d').iloc[0]

            # 파일 저장 경로 및 이름 설정 (obu_id_연월일.csv 형식)
            save_path = os.path.join(processed_path, f"{date_str}_{obu_id}.csv")

            # 새로운 CSV 파일로 저장
            try:
                filtered_df.to_csv(save_path, index=False)
                print(f"파일 저장됨: {save_path}")
            except Exception as e:
                print(f"파일 저장 중 에러 발생: {e}")
    else:
        print(f"'obu_id' 또는 'time' 열이 없음: {file}")
