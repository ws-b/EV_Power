import os
import glob
import pandas as pd
from tqdm import tqdm

# 원본 CSV 파일이 위치한 디렉터리
directory = r'D:\SamsungSTF\Data\KOTI\2019년_팅크웨어_포인트경로데이터'
# 처리된 CSV를 저장할 디렉터리
processed_path = r'D:\SamsungSTF\Processed_Data\KOTI'

# directory 내부의 *.csv 파일 경로 목록을 가져옴
file_lists = glob.glob(os.path.join(directory, "*.csv"))

# tqdm를 사용하여 진행 상황을 표시
for file in tqdm(file_lists):
    # CSV 파일 읽기
    df = pd.read_csv(file)

    # CSV 내에 'obu_id'와 'time' 열이 존재하는지 확인
    if 'obu_id' in df.columns and 'time' in df.columns:
        # 해당 파일에서 중복되지 않는 모든 OBU ID를 가져옴
        obu_ids = df['obu_id'].unique()

        # 파일 내 각 OBU ID별로 데이터 분리 후 처리
        for obu_id in obu_ids:
            # 현재 OBU ID에 해당하는 행만 추출
            obu_df = df[df['obu_id'] == obu_id]

            # 필요한 열('time', 'x', 'y')만 필터링해서 사용
            filtered_df = obu_df[['time', 'x', 'y']].copy()

            # time 컬럼을 datetime으로 변환(포맷이 맞지 않으면 NaT가 됨)
            try:
                filtered_df['time'] = pd.to_datetime(filtered_df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            except ValueError as e:
                print(f"Date format error in file '{file}' for OBU ID: {obu_id}: {e}")
                continue

            # 만약 전체 time이 NaT라면(=변환 실패), 유효한 날짜 형식이 없다고 판단
            if filtered_df['time'].isna().all():
                print(f"파일 '{file}'에서 유효한 날짜 형식을 찾지 못했습니다. OBU ID: {obu_id}")
                continue

            # time 컬럼의 첫 번째 행에서 YYYYMMDD 형태의 날짜 문자열을 추출
            date_str = filtered_df['time'].dt.strftime('%Y%m%d').iloc[0]

            # 추출한 날짜와 OBU ID를 이용해 저장할 CSV 파일 경로 생성
            save_path = os.path.join(processed_path, f"{date_str}_{obu_id}.csv")

            # 이미 동일 파일이 존재한다면 스킵
            if os.path.exists(save_path):
                print(f"파일이 이미 존재합니다: {save_path}. 스킵합니다.")
                continue

            # CSV 파일로 저장
            try:
                filtered_df.to_csv(save_path, index=False)
                print(f"파일 저장됨: {save_path}")
            except Exception as e:
                print(f"파일 저장 중 에러 발생: {e}")
    else:
        # 'obu_id'나 'time' 열이 없으면 스킵 (에러 메시지 표시)
        print(f"'obu_id' 또는 'time' 열이 없음: {file}")
