import os
import numpy as np

def get_file_list(folder_path):
    # 폴더 내의 모든 파일 리스트 가져오기
    file_list = os.listdir(folder_path)
    txt_files = []
    for file in file_list:
        if file.endswith('.csv'):
            txt_files.append(file)
    return txt_files

# 파일이 들어있는 폴더 경로
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 속도-가속도 처리\\'
mac_folder_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터 속도:가속도 처리/'
folder_path = mac_folder_path

# 파일 리스트 가져오기
files = get_file_list(folder_path)
files.sort()

# 파일 리스트 순차적으로 읽기
for file in files:
    # 파일 경로 생성하기
    file_path = os.path.join(folder_path, file)

    # 파일 열기
    with open(file_path, 'r') as f:
        # 파일 읽기
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)
        total_distance = np.cumsum(data[:, 3])
        data = np.column_stack((data, total_distance))
        np.savetxt(file_path, data, delimiter=',', fmt='%f')