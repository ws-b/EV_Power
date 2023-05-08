import os
import numpy as np
import seaborn as sns
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

all_total_distance = []
all_total_time = []

for file in files:
    # 파일 경로 생성하기
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)

    # 시간, 위도, 경도, 속도, 가속도, 총 이동거리, Power 추출
    t, lat, log, v, a, total_distance, Power = data.T

    # 각 파일의 Total distance와 Total time 저장
    all_total_distance.append(total_distance[-1])
    all_total_time.append(t[-1])

# 이동거리(Distance) 히스토그램 그리기
plt.figure()
sns.histplot(all_total_distance, bins='auto', color='blue', kde=True)
plt.xlabel('Total Distance (m)')
plt.ylabel('Number of trips')
plt.title('Total Distance Distribution')
plt.grid(False)
plt.show()

# 총 시간(Total time) 히스토그램 그리기
plt.figure()
sns.histplot(all_total_time, bins='auto', color='green', kde=True)
plt.xlabel('Total Time (s)')
plt.ylabel('Number of trips')
plt.title('Total Time Distribution')
plt.grid(False)
plt.show()
