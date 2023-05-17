import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import math

# 파일이 들어있는 폴더 경로
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'
mac_folder_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터/'

folder_path = win_folder_path

def get_file_list(folder_path):
    # 폴더 내의 모든 파일 리스트 가져오기
    file_list = os.listdir(folder_path)
    csv_files = []
    for file in file_list:
        if file.endswith('.csv'):
            csv_files.append(file)
    return csv_files

files = get_file_list(folder_path)
files.sort()

all_avg_speed_and_mileage = []

for file in files:
    # 파일 경로 생성하기
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)

    # 시간, 위도, 경도, 속도, 가속도, 총 이동거리, Power 추출
    t, lat, log, v, a, total_distance, Power = data.T

    # 전체 Power 합산
    total_power = np.sum(Power)

    # 시간의 총합 계산
    total_time = np.sum(np.diff(t))

    # 평균 속도 계산 (km/h)
    avg_speed = np.mean(v) * 3.6

    # 각 파일의 Total distance / Total Power 계산 (Total Power가 0일 때, 값은 0으로 설정)
    mileage = (total_distance[-1] / 1000) / ((total_power / 1000) * (total_time / 3600)) if total_power != 0 else 0

    # 모든 파일의 avg_speed와 mileage 값 모으기
    all_avg_speed_and_mileage.append((avg_speed, mileage))

# 평균 속도를 10km/h 단위로 범주화
bins = range(0, 160, 10)
binned_avg_speed_and_mileage = [[] for _ in bins]

for avg_speed, mileage in all_avg_speed_and_mileage:
    bin_index = math.floor(avg_speed / 10)
    if bin_index < len(bins):
        binned_avg_speed_and_mileage[bin_index].append(mileage)

# 각 범주별 박스 플롯 그리기
plt.figure(figsize=(10, 6))
sns.boxplot(data=binned_avg_speed_and_mileage)
plt.xlabel('Average Speed (km/h)')
plt.ylabel('Mileage (km/kWh)')
plt.xticks(range(len(bins)), bins)

# x축, y축 범위 제한
plt.ylim(0, 16)
plt.xlim(-0.5, 10.5)

plt.title('Mileage by Average Speed')
plt.show()