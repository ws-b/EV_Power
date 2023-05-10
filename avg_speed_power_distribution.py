import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

all_distance_per_total_power = []
all_average_speed = []

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

    # 각 파일의 Total distance / Total Power 계산 (Total Power가 0일 때, 값은 0으로 설정)
    distance_per_total_power_km_kWh = (total_distance[-1] / 1000) / ((total_power / 1000) * (total_time / 3600)) if total_power != 0 else 0

    # 모든 파일의 distance_per_total_power 값 모으기
    all_distance_per_total_power.append(distance_per_total_power_km_kWh)

    # 각 파일의 평균 속도 저장 # Convert m/s to km/h
    all_average_speed.append(np.mean(v) * 3.6)

# 평균 속도와 전기 주행 거리의 평균 계산
average_speed_mean = np.mean(all_average_speed)
total_distance_per_total_power_mean = np.mean(all_distance_per_total_power)

# 데이터를 2차원 히스토그램으로 변환
hist, xedges, yedges = np.histogram2d(all_average_speed, all_distance_per_total_power, bins=30)

# 히트맵 그리기
sns.heatmap(hist, cmap='viridis', xticklabels=10, yticklabels=5)
# 평균 선 그리기
plt.axvline(average_speed_mean, color='red', linestyle='--')
plt.axhline(total_distance_per_total_power_mean, color='blue', linestyle='--')
# 평균값 표시
plt.text(average_speed_mean + 1, 0.5, f"Mean Speed: {average_speed_mean:.2f} km/h", color='red')
plt.text(1, total_distance_per_total_power_mean + 0.3, f"Mean Electric Mileage: {total_distance_per_total_power_mean:.2f} km/kWh", color='blue')
plt.legend()
plt.xlabel('Average Speed (km/h)')
plt.ylabel('Total Distance / Total Power (km/kWh)')
plt.title('Heatmap of Average Speed vs Electric Mileage')
plt.grid(False)
plt.xlim(0, 130)
plt.ylim(0, 10)
plt.show()