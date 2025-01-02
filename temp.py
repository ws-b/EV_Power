import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 데이터 폴더 경로
data_folder = r'D:\SamsungSTF\Processed_Data\TripByTrip\Highway_Cycle'

# 폴더 내 모든 CSV 파일 목록 가져오기
csv_files = [file for file in os.listdir(data_folder) if file.endswith('.csv')]

# CSV 파일이 2개 이상 있는지 확인
if len(csv_files) < 2:
    raise ValueError("폴더에 CSV 파일이 2개 이상 있어야 합니다.")

# 첫 2개의 CSV 파일 선택
selected_files = csv_files[:2]

# 그래프 저장을 위한 디렉토리 생성
output_dir = os.path.join(data_folder, 'Plots')
os.makedirs(output_dir, exist_ok=True)

n = 1
for file in selected_files:
    file_path = os.path.join(data_folder, file)
    print(f'Processing file: {file_path}')

    # CSV 파일 읽기
    df = pd.read_csv(file_path)

    # 'time' 컬럼을 datetime 형식으로 변환
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

    # 'elapsed_time_min' 컬럼 생성 (첫 시간부터의 경과 시간 in minutes)
    df['elapsed_time_min'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds() / 60

    # 'speed'를 km/h로 변환
    if 'speed' in df.columns:
        df['speed_kmh'] = df['speed'] * 3.6
    else:
        raise ValueError(f"'speed' 컬럼이 {file} 파일에 존재하지 않습니다.")

    # 'Power_data' 컬럼을 kW로 변환
    if 'Power_data' in df.columns:
        df['Power_kW'] = df['Power_data'] / 1000
    else:
        raise ValueError(f"'Power_data' 컬럼이 {file} 파일에 존재하지 않습니다.")

    # 첫 번째 그래프: Speed (km/h) 및 Acceleration with multi y-axis and alpha=0.7
    fig, ax1 = plt.subplots(figsize=(12, 4))

    color1 = 'tab:blue'
    ax1.set_xlabel('Elapsed Time (min)')
    ax1.set_ylabel('Speed (km/h)', color=color1)
    ax1.plot(df['elapsed_time_min'], df['speed_kmh'], color=color1, alpha=0.7, label='Speed (km/h)')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  # 두 번째 y축
    color2 = 'tab:red'
    ax2.set_ylabel('Acceleration', color=color2)
    ax2.plot(df['elapsed_time_min'], df['acceleration'], color=color2, alpha=0.7, label='Acceleration')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 제목 및 레이아웃 설정
    plt.title(f'Speed and Acceleration (Highway cycle {n})')
    fig.tight_layout()

    # 그래프 저장
    plot1_path = os.path.join(output_dir, f'{os.path.splitext(file)[0]}_speed_acceleration.png')
    plt.savefig(plot1_path)
    plt.close()
    print(f'Saved plot: {plot1_path}')

    # 두 번째 그래프: Power (kW) (색상 변경 및 레이블 수정)
    plt.figure(figsize=(12, 4))

    if 'Power_kW' in df.columns:
        plt.plot(df['elapsed_time_min'], df['Power_kW'], color='tab:green', alpha=0.7, label='Power (kW)')
    else:
        raise ValueError(f"'Power_kW' 컬럼이 {file} 파일에 존재하지 않습니다.")

    plt.xlabel('Elapsed Time (min)')
    plt.ylabel('Power (kW)', color='tab:green')
    plt.title(f'Power Over Time (Highway cycle {n})')
    plt.tick_params(axis='y', labelcolor='tab:green')
    plt.tight_layout()

    # 그래프 저장
    plot2_path = os.path.join(output_dir, f'{os.path.splitext(file)[0]}_power.png')
    plt.savefig(plot2_path)
    plt.close()
    print(f'Saved plot: {plot2_path}')

    n += 1

print("모든 그래프 생성이 완료되었습니다.")
