import os
import pandas as pd
from tqdm import tqdm

start_path = '/Volumes/Data/test_case/'  # 시작 디렉토리

def get_year_month_from_filename(file_name):
    """ 파일명에서 연월 추출 """
    try:
        parts = file_name.split('_')
        year_month = parts[2][:7]  # 연월 (YYYY-MM 형식)
        return year_month
    except IndexError:
        return None

total_folders = sum([len(dirs) == 0 for _, dirs, _ in os.walk(start_path)])

with tqdm(total=total_folders, desc="진행 상황", unit="folder") as pbar:
    for root, dirs, files in os.walk(start_path):
        if not dirs:
            bms_files = [f for f in files if f.startswith('bms_') and f.endswith('.csv')]
            altitude_files = [f for f in files if f.startswith('altitude_') and f.endswith('.csv')]

            for bms_file in bms_files:
                bms_year_month = get_year_month_from_filename(bms_file)
                bms_path = os.path.join(root, bms_file)
                bms_df = pd.read_csv(bms_path, encoding='utf-8')
                bms_df['time'] = pd.to_datetime(bms_df['time'], format='%Y-%m-%d %H:%M:%S')

                for altitude_file in altitude_files:
                    altitude_year_month = get_year_month_from_filename(altitude_file)

                    if bms_year_month == altitude_year_month:
                        altitude_path = os.path.join(root, altitude_file)
                        altitude_df = pd.read_csv(altitude_path, usecols=['time', 'altitude'], encoding='utf-8')
                        altitude_df['time'] = pd.to_datetime(altitude_df['time'], format='%Y-%m-%d %H:%M:%S')

                        # 가장 가까운 시간을 기준으로 병합
                        merged_df = pd.merge_asof(bms_df.sort_values('time'), altitude_df.sort_values('time'), on='time', by='device_no', tolerance=pd.Timedelta('1min'), direction='nearest')

                        # 병합된 데이터 저장
                        output_file_name = f'merged_bms_altitude_{bms_year_month}.csv'
                        output_path = os.path.join(root, output_file_name)
                        merged_df.to_csv(output_path, index=False)
                        break

            pbar.update(1)

print("모든 폴더의 파일 처리가 완료되었습니다.")
