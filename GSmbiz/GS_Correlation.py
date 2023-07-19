import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm


win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\ioniq5'

folder_path = os.path.normpath(win_folder_path)

# get a list of all .csv files in the folder
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# for file in tqdm(file_lists[20:21]):
#     file_path = os.path.join(folder_path, file)
#     data = pd.read_csv(file_path)
#
#     # 열 제외
#     columns_to_drop = ['time', 'chrg_cable_conn', 'soh', 'cell_volt_list', 'pack_current', 'pack_volt', 'Energy']
#
#     # 상관 계수 계산
#     correlation_matrix = data.corr()
#
#     # 히트맵 생성
#     plt.figure(figsize=(12, 10))  # 히트맵의 사이즈를 조절합니다.
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True,
#                 cbar_kws={"shrink": .8}, linewidths=.5)  # 상관계수를 소수점 둘째자리까지 표시합니다.
#
#     # 그래프 출력
#     plt.xticks(rotation=30)  # x축 레이블의 각도를 조정하여 잘리는 것을 방지합니다.
#     plt.yticks(rotation=30)  # y축 레이블의 각도를 조정하여 잘리는 것을 방지합니다.
#     plt.subplots_adjust(left=0.15, right=0.98, top=0.98)  # 마진을 조정합니다.
#     plt.show()


# Variables of interest
vars_of_interest = ['emobility_spd_m_per_s', 'acceleration', 'trip_chrg_pw', 'trip_dischrg_pw', 'ext_temp', 'int_temp', 'soc']

for file in tqdm(file_lists[20:21]):
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)

    # Only keep the columns of interest
    data = data[vars_of_interest]

    # Calculate rolling correlations
    rolling_correlation = data.rolling(window=60).corr()  # change window size as necessary

    # Plot each rolling correlation
    for var in vars_of_interest:
        plt.figure(figsize=(12, 10))
        plt.title(f'Rolling Correlation with {var}')
        rolling_correlation[var].dropna().plot()
        plt.xticks(rotation=30)
        plt.show()
