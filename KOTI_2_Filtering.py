import pandas as pd
import os
from tqdm import tqdm

# 경로 설정
processed_path = r'D:\SamsungSTF\Processed_Data\KOTI'  # 처리된 CSV 파일이 위치한 폴더 경로
invalid_year_path = os.path.join(processed_path, 'invalid_year')  # 잘못된 연도가 발견될 경우 파일을 옮길 폴더 (혹은 별도로 쓰일 폴더)

# 필요한 폴더가 없으면 생성
os.makedirs(invalid_year_path, exist_ok=True)

# 지정된 경로에서 CSV 파일 목록 가져오기
csv_files = [f for f in os.listdir(processed_path) if f.endswith('.csv')]

# 모든 CSV 파일 처리
for file in tqdm(csv_files, desc="Processing CSV files"):
    file_path = os.path.join(processed_path, file)

    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
    except Exception as e:
        # 파일 읽기 에러 발생 시 알림
        print(f"Error reading {file}: {e}")
        continue

    # 데이터가 없는 파일(컬럼명만 존재하거나 완전히 빈 파일) 삭제
    if df.empty:
        try:
            os.remove(file_path)
            print(f"Deleted empty file: {file}")
        except Exception as e:
            print(f"Error deleting empty file {file}: {e}")
        # 다음 파일 처리로 넘어감
        continue

    # 행 수가 2개 이하인 경우(헤더 포함, 실제 데이터 부족) 삭제
    if len(df) <= 2:
        try:
            os.remove(file_path)
            print(f"Deleted file with <=2 rows: {file}")
        except Exception as e:
            print(f"Error deleting file with <=2 rows {file}: {e}")
        # 다음 파일 처리로 넘어감
        continue

    # 'time' 열을 datetime 형태로 변환
    # 포맷이 '%Y-%m-%d %H:%M:%S'와 다르거나 변환이 불가능할 경우 NaT(Not a Time)로 표시
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # 잘못된 연도(2018, 2019, 2020년 외)나 NaT가 있는 행 찾기
    # -> df['time'].isna()는 NaT 행을 찾아냄
    invalid_years = df[~df['time'].dt.year.isin([2018, 2019, 2020]) | df['time'].isna()]

    # 잘못된 연도가 포함된 파일은 삭제 (혹은 필요한 경우 invalid_year_path로 옮길 수도 있음)
    if not invalid_years.empty:
        try:
            os.remove(file_path)
            print(f"Deleted file with invalid years: {file}")
        except Exception as e:
            print(f"Error deleting file with invalid years {file}: {e}")
        continue

print("Processing and file moving completed.")
