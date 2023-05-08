import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 파일이 들어있는 폴더 경로
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 Processed\\'
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

departure_minutes = []

for file in files:
    # 파일 경로 생성하기
    file_path = os.path.join(folder_path, file)
    T = np.loadtxt(file_path, delimiter=',', dtype=str, usecols=0)

    # 첫 번째 T값 (출발 시각) 가져오기
    departure_time_str = T[0]

    # datetime 객체로 변환
    departure_time = datetime.strptime(departure_time_str, "%Y-%m-%d %H:%M:%S")

    # 출발 시각의 시간:분 값 저장 (0부터 1439까지의 정수값)
    departure_minutes.append(departure_time.hour * 60 + departure_time.minute)

# 출발 시각의 분포를 히스토그램으로 그리기
plt.figure()
sns.histplot(departure_minutes, bins=range(0, 1441, 60), color='purple', kde=False)
plt.xlabel('Departure Time (HH:MM)')
plt.ylabel('Number of trips')
plt.title('Departure Time Distribution')
plt.xticks(range(0, 1440, 60), [f"{h:02d}:00" for h in range(0, 24)], rotation=45)
plt.grid(False)
plt.show()