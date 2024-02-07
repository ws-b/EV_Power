import os
import pandas as pd
from tqdm import tqdm
import chardet

start_path = '/Users/wsong/Downloads/test_case/'  # 시작 디렉토리

def extract_info_from_filename(file_name):
    """파일명에서 단말기 번호와 연월 추출"""
    try:
        parts = file_name.split('_')
        device_no = parts[2]  # 단말기 번호
        date_parts = parts[3].split('-')
        year_month = '-'.join(date_parts[:2])  # 연월 (YYYY-MM 형식)
        return device_no, year_month
    except IndexError:
        return None, None

def read_file_with_detected_encoding(file_path):
    # 파일의 인코딩을 감지하여 데이터를 읽음
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))  # 첫 100,000 바이트를 사용하여 인코딩 감지
    encoding = result['encoding']
    return pd.read_csv(file_path, encoding=encoding, header=0)

total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)])

with tqdm(total=total_folders, desc="진행 상황", unit="folder") as pbar:
    for root, dirs, files in os.walk(start_path):
        if not dirs:
            filtered_files = [file for file in files if 'bms' in file and file.endswith('.csv')]
            filtered_files.sort()
            dfs = []
            device_no, year_month = None, None
            for file in filtered_files:
                file_path = os.path.join(root, file)
                df = read_file_with_detected_encoding(file_path)

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
                # 현재 폴더의 상위 폴더에 파일 저장
                parent_folder = os.path.dirname(root)
                combined_df.to_csv(os.path.join(parent_folder, output_file_name), index=False)

            pbar.update(1)

print("모든 폴더의 파일 처리가 완료되었습니다.")
