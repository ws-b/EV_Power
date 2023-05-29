import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 파일이 들어있는 폴더 경로
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\kona_ev\\'
mac_folder_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터 속도-가속도 처리'

folder_path = win_folder_path

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

all_distance_per_total_power = []
over_30_files = []

for file_list in file_lists:
    # 파일 경로 생성하기
    file_path = os.path.join(folder_path, file_list)
    data = pd.read_csv(file_path)

    # 시간, 속도, Power 추출
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    v = data['emobility_spd_m_per_s']
    Power = data['Power']

    # 샘플링 간격 (2초)를 고려하여 총 이동거리 계산
    total_distance = np.sum(v * 2)

    # 전체 Power 합산
    total_power = np.sum(Power)

    # 시간의 총합 계산
    total_time = np.sum(t.diff().dt.total_seconds())

    # 각 파일의 Total distance / Total Power 계산 (Total Power가 0일 때, 값은 0으로 설정)
    distance_per_total_power_km_kWh = (total_distance / 1000) / ((total_power / 1000) * (total_time / 3600)) if total_power != 0 else 0

    # 모든 파일의 distance_per_total_power 값 모으기
    all_distance_per_total_power.append(distance_per_total_power_km_kWh)

    # 만약 distance_per_total_power_km_kWh가 50 이상이면, 해당 파일 이름을 over_50_files에 추가
    if distance_per_total_power_km_kWh >= 30:
        over_30_files.append(file_list)

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

# 50 이상의 비율을 가진 파일들 출력
print("Files with a ratio of Total Distance / Total Power greater than 50:")
for file in over_30_files:
    print(file)