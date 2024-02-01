import os
import pandas as pd
from tqdm import tqdm

start_path = '/Volumes/Data/test_case/'  # 시작 디렉토리

def extract_info_from_filename(file_name):
    """ 파일명에서 단말기 번호와 연월 추출 """
    try:
        parts = file_name.split('_')
        device_no = parts[2]  # 단말기 번호
        date_parts = parts[3].split('-')
        year_month = '-'.join(date_parts[:2])  # 연월 (YYYY-MM 형식)
        return device_no, year_month
    except IndexError:
        return None, None

total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)])

with tqdm(total=total_folders, desc="진행 상황", unit="folder") as pbar:
    for root, dirs, files in os.walk(start_path):
        if not dirs:
            filtered_files = [file for file in files if 'bms' in file and file.endswith('.csv')]
            filtered_files.sort()  # 파일 이름으로 정렬
            dfs = []  # 각 파일의 DataFrame을 저장할 리스트
            device_no, year_month = None, None
            for file in filtered_files:
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path, header=0, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, header=0, encoding='cp949')

                # 'Unnamed'으로 시작하는 컬럼 제거
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

                # 첫 행을 제외하고 역순으로 정렬
                df = df.iloc[1:][::-1]
                dfs.append(df)

                if device_no is None or year_month is None:
                    device_no, year_month = extract_info_from_filename(file)

            if dfs and device_no and year_month:
                combined_df = pd.concat(dfs, ignore_index=True)
                output_file_name = f'bms_{device_no}_{year_month}.csv'
                combined_df.to_csv(os.path.join(root, output_file_name), index=False)

            pbar.update(1)

print("모든 최하위 폴더의 파일 처리가 완료되었습니다.")
