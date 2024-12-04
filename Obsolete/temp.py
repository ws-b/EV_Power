import pandas as pd
import glob
import os

# 폴더 경로 설정
folder_path = r'C:\Users\BSL\Desktop\DRT'

# 처리할 열 목록
columns_to_keep = [
    'time', 'speed', 'acceleration', 'ext_temp',
    'mod_temp_list', 'soc', 'soh',
    'cell_volt_list', 'pack_volt', 'pack_current'
]

# 폴더 내 모든 CSV 파일 찾기
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# 각 CSV 파일 처리
for file in csv_files:
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file)

        # 필요한 열만 선택
        df = df[columns_to_keep]

        # 'time' 컬럼을 datetime 형식으로 변환
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

        # 첫 번째 시간 저장
        start_time = df['time'].iloc[0]

        # 경과 시간을 초 단위로 계산
        df['time'] = (df['time'] - start_time).dt.total_seconds()

        # 첫 번째 행의 'time'을 0으로 설정 (이미 계산된 상태)
        # 만약 누락된 값이 있을 경우, 첫 번째 유효한 값을 0으로 설정
        df['time'] = df['time'] - df['time'].iloc[0]

        # 처리된 데이터를 원본 CSV 파일에 덮어쓰기
        df.to_csv(file, index=False)

        print(f"Processed and overwritten: {file}")

    except Exception as e:
        print(f"Error processing {file}: {e}")
