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
    data = pd.read_csv(folder_path + file_list)

    # 시간, 속도, 가속도, Power 추출
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

    # The first time in the time column set as reference (0 seconds
    t = (t - t.iloc[0]).dt.total_seconds().tolist()

    v = data['emobility_spd_m_per_s'].tolist()
    a = data['acceleration'].tolist()
    Power = data['Power'].tolist()

    # 총 이동거리 계산 (속도는 m/s 단위로 가정합니다.)
    total_distance = sum(v) * 2 / 1000  # 시간 간격(2초)와 속도를 m에서 km로 변환

    # 전체 Power 합산
    total_power = sum(Power)

    # 각 파일의 Total distance / Total Power 계산 (Total Power가 0일 때, 값은 0으로 설정)
    distance_per_total_power_km_kWh = total_distance / total_power if total_power != 0 else 0

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
plt.xlim(0, 25)
plt.xlabel('Total Distance / Total Power (km/kWh)')
plt.ylabel('Number of trips')
plt.title('Total Distance / Total Power Distribution')
plt.grid(False)
plt.show()