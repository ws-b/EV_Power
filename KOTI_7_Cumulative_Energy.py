import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid

# CSV 파일들이 위치한 디렉토리 설정
csv_folder = r"D:\SamsungSTF\Processed_Data\KOTI"

# 처리할 CSV 파일 리스트 가져오기
file_paths = [os.path.join(csv_folder, file) for file in os.listdir(csv_folder) if file.endswith('.csv')]

if not file_paths:
    print(f"지정된 폴더에 CSV 파일이 없습니다: {csv_folder}")
    exit()

# 트립 요약 정보를 저장할 리스트 초기화
trip_summaries = []

# Loop through all CSV files in the directory
for file_path in tqdm(file_paths):
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # 시간 순으로 정렬
        df = df.sort_values('time').reset_index(drop=True)

        # 시작 시점부터의 시간(초) 계산
        time_seconds = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

        # cumulative_trapezoid를 사용하여 누적 에너지 계산
        cum_energy = cumulative_trapezoid(y=df['Power_hybrid'], x=time_seconds, initial=0)

        # 누적 에너지 컬럼 추가
        df['cumul_energy'] = cum_energy/3600/1000

        # 총 에너지 소비량 계산
        total_energy = df['cumul_energy'].iloc[-1]

        # 평균 속도 계산
        if 'speed' in df.columns:
            average_speed = df['speed'].mean()
        else:
            average_speed = np.nan
            print(f"'speed' 컬럼이 없습니다: {file_path}")

        # 급가속/급감속 기준 설정
        rapid_acc_threshold = 5.0  # m/s² 이상 급가속
        rapid_dec_threshold = -5.0  # m/s² 이하 급감속

        # 급가속/급감속 이벤트 카운트
        rapid_acc_count = (df['acceleration'] > rapid_acc_threshold).sum()
        rapid_dec_count = (df['acceleration'] < rapid_dec_threshold).sum()

        # 트립 전체 시간(분) 계산
        trip_duration_seconds = (df['time'].max() - df['time'].min()).total_seconds()
        trip_duration_minutes = trip_duration_seconds / 60 if trip_duration_seconds > 0 else np.nan

        # 급가속/급감속 이벤트 수를 분당으로 변환
        if not np.isnan(trip_duration_minutes) and trip_duration_minutes > 0:
            rapid_acc_rate = rapid_acc_count / trip_duration_minutes
            rapid_dec_rate = rapid_dec_count / trip_duration_minutes
        else:
            rapid_acc_rate = np.nan
            rapid_dec_rate = np.nan
            print(f"트립 시간 계산에 문제가 있습니다: {file_path}")

        # 트립 요약 정보 생성
        trip_summary = {
            'Trip_ID': os.path.basename(file_path),
            'Total_Time[sec]': trip_duration_seconds,
            'Total_Energy[kWh]': total_energy,
            'Average_Speed[km/h]': average_speed*3.6,
            'Rapid_Acc_Rate_per_min[m/s^2]': rapid_acc_rate,
            'Rapid_Dec_Rate_per_min[m/s^2]': rapid_dec_rate
        }

        trip_summaries.append(trip_summary)

        # 수정된 DataFrame을 원본 CSV에 저장
        df.to_csv(file_path, index=False)

        print(f"처리 완료: {file_path}")

    except Exception as e:
        print(f"파일 처리 중 오류 발생 ({file_path}): {e}")

# 트립 요약 정보를 DataFrame으로 변환
summary_df = pd.DataFrame(trip_summaries)

# 요약 정보를 지정된 경로에 CSV로 저장
output_path = r"C:\Users\BSL\Desktop\trip_summary.csv"
try:
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"트립 요약 정보가 성공적으로 저장되었습니다: {output_path}")
except Exception as e:
    print(f"트립 요약 정보 저장 중 오류 발생: {e}")
