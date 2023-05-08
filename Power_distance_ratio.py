import os
import numpy as np

# 파일이 들어있는 폴더 경로
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'
mac_folder_path = '/Users/woojin/Desktop/대학교 자료/켄텍 자료/삼성미래과제/Driving Pattern/Drive Cycle Processed/'

folder_path = mac_folder_path

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

distance_per_total_power = []

for file in files:
    # 파일 경로 생성하기
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)

    # 시간, 속도, 가속도, 총 이동거리, Power 추출
    t, v, a, total_distance, Power = data.T

    # 전체 Power 합산
    total_power = np.sum(Power)

    ratio = data[-1, 3] / total_power
    print("Ratio : ", ratio)