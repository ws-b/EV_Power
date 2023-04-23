import os
import numpy as np
import matplotlib.pyplot as plt

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

all_distance_per_power = []

for file in files:
    # 파일 경로 생성하기
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)

    # 시간, 위도, 경도, 속도, 가속도, 총 이동거리, Power 추출
    t, lat, log, v, a, total_distance, Power = data.T

    # Total distance / Power 계산
    distance_per_power = [total_distance[i] / Power[i] for i in range(len(Power))]

    # 모든 파일의 distance_per_power 값 모으기
    all_distance_per_power.extend(distance_per_power)

# 전체 파일에 대한 히스토그램 그리기
plt.hist(all_distance_per_power, bins='auto', alpha=0.7, color='blue')
plt.xlabel('Total Distance / Power (m/kW)')
plt.ylabel('Frequency')
plt.title('Total Distance / Power Distribution for All Files')
plt.grid(True)
plt.show()
