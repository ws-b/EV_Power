import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# 파일이 들어있는 폴더 경로
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\Ioniq5\\'
mac_folder_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터 속도-가속도 처리'

folder_path = win_folder_path

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_distance_per_total_power = []

for file_list in file_lists:
    # 파일 경로 생성하기
    file_path = os.path.join(folder_path, file_list)
    data = pd.read_csv(file_path)

    # 시간, 속도, CHARGE, DISCHARGE 추출
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    v = data['emobility_spd_m_per_s']
    CHARGE = data['trip_chrg_pw']
    DISCHARGE = data['trip_dischrg_pw']

    # 샘플링 간격 (2초)를 고려하여 총 이동거리 계산
    total_distance = np.sum(v * 2)

    # DISCHARGE 합산에서 CHARGE 합산을 빼기
    net_discharge = np.sum(DISCHARGE) - np.sum(CHARGE)

    # 시간의 총합 계산
    total_time = np.sum(t.diff().dt.total_seconds())

    # 각 파일의 Total distance / net_discharge 계산 (net_discharge가 0일 때, 값은 0으로 설정)
    distance_per_total_power_km_kWh = (total_distance / 1000) / ((net_discharge ) ) if net_discharge != 0 else 0

    # 모든 파일의 distance_per_total_power 값 모으기
    all_distance_per_total_power.append(distance_per_total_power_km_kWh)

# 전체 파일에 대한 히스토그램 그리기
hist_data = sns.histplot(all_distance_per_total_power, bins='auto', color='gray', kde=False)

# 평균 세로선 그리기
mean_value = np.mean(all_distance_per_total_power)
plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')

# 평균값 표시
plt.text(mean_value + 0.05, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='red', fontsize=12)

# x축 범위 설정 (0부터 25까지)
#plt.xlim(0, 25)
plt.xlabel('Total Distance / Net Discharge (km/kWh)')
plt.ylabel('Number of trips')
plt.title('Total Distance / Net Discharge Distribution')
plt.grid(False)
plt.show()