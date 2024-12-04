import os
import pandas as pd
from glob import glob
from datetime import datetime
import shutil
import math
from tqdm import tqdm
import numpy as np

# 분석할 디렉토리 경로
directory_path = r'D:\SamsungSTF\Processed_Data\KOTI'

# 모든 CSV 파일 경로 가져오기
csv_files = glob(os.path.join(directory_path, '*.csv'))
total_files = len(csv_files)

# 저장할 경로 설정
save_directory = r'C:\Users\BSL\Desktop'
os.makedirs(save_directory, exist_ok=True)

# 필터링된 파일을 이동할 디렉토리 설정
filtered_directory = r'D:\SamsungSTF\Processed_Data\KOTI\Filtered'
os.makedirs(filtered_directory, exist_ok=True) 

# 결과를 저장할 리스트
summary_data = []

# O가 하나라도 들어간 파일 갯수를 세기 위한 변수
files_with_any_O_count = 0
files_with_O_in_2_to_6_count = 0

# 조건 2~6에서 O가 하나라도 들어간 파일명을 저장할 리스트
files_with_O_in_2_to_6 = []

# tqdm을 사용하여 진행률 표시
for file in tqdm(csv_files, desc="Processing CSV files", unit="file"):
    try:
        df = pd.read_csv(file)
        filename = os.path.basename(file)

        # 결과 저장을 위한 딕셔너리 초기화
        conditions = {
            'Filename': filename,
            'Condition1': 'X',
            'Condition2': 'X',
            'Condition3': 'X',
            'Condition4': 'X',
            'Condition5': 'X',
            'Condition6': 'X'  # Condition6 추가
        }

        # 'speed' 컬럼이 존재하는지 확인
        if 'speed' not in df.columns:
            print(f"\n파일 {file}에 'speed' 컬럼이 없습니다. 건너뜁니다.")
            continue

        # Condition 1: 첫 두 행의 'speed'가 0인지 확인
        if len(df) >= 2:
            if not (df['speed'].iloc[0] == 0 and df['speed'].iloc[1] == 0):
                conditions['Condition1'] = 'O'
            else:
                conditions['Condition1'] = 'X'
        else:
            # 데이터가 2행 미만인 경우
            conditions['Condition1'] = 'X'

        # Condition 2: 'acceleration' 컬럼의 절대값이 9.8을 넘는지 확인
        if 'acceleration' in df.columns:
            acceleration_abs = df['acceleration'].abs()
            if (acceleration_abs > 9.8).any():
                conditions['Condition2'] = 'O'
            else:
                conditions['Condition2'] = 'X'
        else:
            # 'acceleration' 컬럼이 없는 경우
            conditions['Condition2'] = 'X'

        # Condition 3: 평균 속도가 110을 넘는지 확인
        avg_speed = df['speed'].mean()
        if avg_speed > 110/3.6:
            conditions['Condition3'] = 'O'
        else:
            conditions['Condition3'] = 'X'

        # Condition 4: 'time' 컬럼의 첫 행과 끝 행의 차이가 600초를 넘지 않는지 확인
        if 'time' in df.columns:
            # 'time' 컬럼을 문자열로 변환
            df['time'] = df['time'].astype(str)

            # 첫 행과 끝 행의 'time' 값을 가져오기
            start_time_str = df['time'].iloc[0]
            end_time_str = df['time'].iloc[-1]

            # 시간 형식 변환
            time_format = '%Y-%m-%d %H:%M:%S'
            try:
                start_time = datetime.strptime(start_time_str, time_format)
                end_time = datetime.strptime(end_time_str, time_format)
            except ValueError as ve:
                print(f"\n파일 {file}의 'time' 형식이 올바르지 않습니다: {ve}. 건너뜁니다.")
                continue

            # 시간 차이 계산 (초 단위)
            time_diff = (end_time - start_time).total_seconds()

            if time_diff <= 600:
                conditions['Condition4'] = 'O'
            else:
                conditions['Condition4'] = 'X'
        else:
            print(f"\n파일 {file}에 'time' 컬럼이 없습니다. 건너뜁니다.")
            continue

        # Condition 5: speed를 time으로 적분해서 1000m가 안되면 O
        # 먼저 time을 datetime으로 변환했으므로, time을 초 단위로 변환
        time_in_seconds = df['time'].apply(lambda x: datetime.strptime(x, time_format).timestamp())
        time_in_seconds = time_in_seconds - time_in_seconds.iloc[0]  # 첫 시간으로부터의 상대 시간 (초)

        # 속도를 m/s로 유지하여 적분하면 결과는 미터 단위
        # 적분 수행
        try:
            distance = np.trapz(y=df['speed'], x=time_in_seconds)
            if distance < 1000:
                conditions['Condition5'] = 'O'
            else:
                conditions['Condition5'] = 'X'
        except Exception as e:
            print(f"\n파일 {file}에서 적분 중 오류 발생: {e}. 건너뜁니다.")
            continue

        # Condition 6: 'ext_temp' 컬럼에서 빈 항목이 하나라도 있는지 확인
        if 'ext_temp' in df.columns:
            if df['ext_temp'].isnull().any() or (df['ext_temp'] == '').any():
                conditions['Condition6'] = 'O'
            else:
                conditions['Condition6'] = 'X'
        else:
            # 'ext_temp' 컬럼이 없는 경우
            conditions['Condition6'] = 'O'

        # 조건들 중 하나라도 'O'가 있는지 확인
        if 'O' in [
            conditions['Condition1'],
            conditions['Condition2'],
            conditions['Condition3'],
            conditions['Condition4'],
            conditions['Condition5'],
            conditions['Condition6']  # Condition6 포함
        ]:
            files_with_any_O_count += 1

        # 조건 2~6 중 하나라도 'O'가 있는지 확인
        if 'O' in [
            conditions['Condition2'],
            conditions['Condition3'],
            conditions['Condition4'],
            conditions['Condition5'],
            conditions['Condition6']  # Condition6 포함
        ]:
            files_with_O_in_2_to_6_count += 1
            files_with_O_in_2_to_6.append(filename)

        # 결과를 리스트에 추가
        summary_data.append(conditions)

    except Exception as e:
        print(f"\n파일 {file} 처리 중 오류 발생: {e}")

# DataFrame 생성
summary_df = pd.DataFrame(summary_data)

# 전체 파일 수
total_files = len(summary_df)

# 조건별로 'O'의 개수 세기
condition_counts = {}
for i in range(1, 7):  # 1부터 6까지
    condition_counts[f'Condition{i}'] = (summary_df[f'Condition{i}'] == 'O').sum()

# 전체 파일 중 하나라도 'O'가 있는 파일 수 출력
print(f"\nO가 하나라도 있는 파일의 개수: {files_with_any_O_count}개")

# 조건 2~6 중 하나라도 'O'인 파일 수 출력
print(f"\nCondition 2~6 중 하나라도 O인 파일의 개수: {files_with_O_in_2_to_6_count}개")

# 조건 2~6 중 하나라도 'O'인 파일 목록 저장
if len(files_with_O_in_2_to_6) > 0:
    condition_2_6_file_path = os.path.join(save_directory, 'files_with_O_in_conditions_2_to_6.txt')
    with open(condition_2_6_file_path, 'w', encoding='utf-8') as f:
        for filename in files_with_O_in_2_to_6:
            f.write(f"{filename}\n")
    print(f"Condition 2~6 중 하나라도 O인 파일 목록이 '{condition_2_6_file_path}'에 저장되었습니다.")

    # 파일들을 지정된 'Filtered' 디렉토리로 이동
    for filename in tqdm(files_with_O_in_2_to_6, desc="Moving Filtered Files", unit="file"):
        source_path = os.path.join(directory_path, filename)
        destination_path = os.path.join(filtered_directory, filename)
        try:
            shutil.move(source_path, destination_path)
            # 또는 복사 후 삭제하고 싶다면 shutil.copy2 + os.remove 사용
        except Exception as e:
            print(f"파일 이동 중 오류 발생: {filename} -> {e}")

    print(f"조건 2~6을 만족하는 파일들이 '{filtered_directory}'로 이동되었습니다.")

# 각 조건별로 'O'의 개수 출력
print("\n각 조건별로 'O'의 개수:")
for i in range(1, 7):
    print(f"Condition {i}: {condition_counts[f'Condition{i}']}개")
