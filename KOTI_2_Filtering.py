import pandas as pd
import os
from tqdm import tqdm

# 경로 설정
processed_path = r'D:\SamsungSTF\Processed_Data\KOTI'
invalid_year_path = os.path.join(processed_path, 'invalid_year')

# 필요한 폴더가 없으면 생성
os.makedirs(invalid_year_path, exist_ok=True)

# 파일 로드
csv_files = [f for f in os.listdir(processed_path) if f.endswith('.csv')]

# 모든 CSV 파일 처리
for file in tqdm(csv_files, desc="Processing CSV files"):
    file_path = os.path.join(processed_path, file)

    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

    # 데이터가 없는 파일(컬럼명만 있는 경우) 삭제
    if df.empty:
        try:
            os.remove(file_path)
            print(f"Deleted empty file: {file}")
        except Exception as e:
            print(f"Error deleting empty file {file}: {e}")
        continue  # 다음 파일로 넘어감

    # 행 수가 2개 이하인 파일 삭제
    if len(df) <= 2:
        try:
            os.remove(file_path)
            print(f"Deleted file with <=2 rows: {file}")
        except Exception as e:
            print(f"Error deleting file with <=2 rows {file}: {e}")
        continue  # 다음 파일로 넘어감

    # time 열을 datetime으로 변환
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # 잘못된 연도 찾기 (2018, 2019, 2020년 외의 값 확인)
    # 'time' 컬럼에 NaT가 있는 경우도 제외
    invalid_years = df[~df['time'].dt.year.isin([2018, 2019, 2020]) | df['time'].isna()]

    if not invalid_years.empty:
        try:
            os.remove(file_path)
            print(f"Deleted file with invalid years: {file}")
        except Exception as e:
            print(f"Error deleting file with invalid years {file}: {e}")
        continue

print("Processing and file moving completed.")
