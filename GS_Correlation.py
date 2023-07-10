import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\ioniq5'
mac_folder_path = ''

folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

for file in file_lists[20:21]:
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # 'time' 열 제외
    data = data.drop(columns='time')
    data = data.drop(columns='chrg_cable_conn')
    data = data.drop(columns='soh')

    # 상관 계수 계산
    correlation_matrix = data.corr()

    # 히트맵 생성
    plt.figure(figsize=(12, 10))  # 히트맵의 사이즈를 조절합니다.
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True,
                cbar_kws={"shrink": .8}, linewidths=.5)  # 상관계수를 소수점 둘째자리까지 표시합니다.

    # 그래프 출력
    plt.xticks(rotation=30)  # x축 레이블의 각도를 조정하여 잘리는 것을 방지합니다.
    plt.yticks(rotation=30)  # y축 레이블의 각도를 조정하여 잘리는 것을 방지합니다.
    plt.subplots_adjust(left=0.15, right=0.98, top=0.98)  # 마진을 조정합니다.
    plt.show()