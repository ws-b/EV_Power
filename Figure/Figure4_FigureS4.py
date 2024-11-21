import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 저장할 경로 설정
fig_save_path = r"C:\Users\BSL\Desktop\Figures\Supplementary"


def figure4(city_cycle1, highway_cycle1, city_cycle2, highway_cycle2, fig_number):
    fig, axs = plt.subplots(2, 4, figsize=(24, 10))

    scaling = 1.4
    # 폰트 크기 설정
    plt.rcParams['font.size'] = 10 * scaling  # 기본 폰트 크기
    plt.rcParams['axes.titlesize'] = 12 * scaling  # 제목 폰트 크기
    plt.rcParams['axes.labelsize'] = 10 * scaling  # 축 레이블 폰트 크기
    plt.rcParams['xtick.labelsize'] = 10 * scaling  # X축 틱 레이블 폰트 크기
    plt.rcParams['ytick.labelsize'] = 10 * scaling  # Y축 틱 레이블 폰트 크기
    plt.rcParams['legend.fontsize'] = 10 * scaling  # 범례 폰트 크기
    plt.rcParams['legend.title_fontsize'] = 10 * scaling  # 범례 제목 폰트 크기
    plt.rcParams['figure.titlesize'] = 12 * scaling  # 그림 제목 폰트 크기

    def process_and_plot_power(file, ax, marker, title):
        data = pd.read_csv(file)

        # 시간 처리
        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff)
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60
        date = t.iloc[0].strftime('%Y-%m-%d')

        # 전력 데이터
        power_data = np.array(data['Power_data']) / 1000  # kW로 변환
        power_phys = np.array(data['Power_phys']) / 1000  # kW로 변환

        # 하이브리드 전력 (있다면)
        if 'Power_hybrid' in data.columns:
            power_hybrid = np.array(data['Power_hybrid']) / 1000

        # 그래프 그리기
        ax.set_xlabel('Time (minutes)', fontsize=10 * scaling)
        ax.set_ylabel('Power (kW)', fontsize=10 * scaling)
        ax.plot(t_min, power_data, label='Data', color='tab:blue', alpha=0.6)
        ax.plot(t_min, power_phys, label='Physics Model', color='tab:red', alpha=0.6)

        if 'Power_hybrid' in data.columns:
            ax.plot(t_min, power_hybrid, label='Hybrid Model', color='tab:green', alpha=0.6)

        ax.legend(loc='upper left', bbox_to_anchor=(0, 0.99))
        ax.set_title(title, pad=10)
        ax.tick_params(axis='both', which='major', labelsize=10 * scaling)
        ax.text(-0.1, 1.05, marker, transform=ax.transAxes, size=16 * scaling, weight='bold', ha='left')  # 마커 추가

    def process_and_plot_energy(file, ax, marker, title):
        data = pd.read_csv(file)

        # 시간 처리
        t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        t_diff = t.diff().dt.total_seconds().fillna(0)
        t_diff = np.array(t_diff)
        t_min = (t - t.iloc[0]).dt.total_seconds() / 60  # 분 단위로 변환

        power_data = np.array(data['Power_data'])
        energy_data = power_data * t_diff / 3600 / 1000
        energy_data_cumulative = energy_data.cumsum()

        if 'Power_phys' in data.columns:
            power_phys = np.array(data['Power_phys'])
            energy_phys = power_phys * t_diff / 3600 / 1000
            energy_phys_cumulative = energy_phys.cumsum()

        if 'Power_hybrid' in data.columns:
            power_hybrid = np.array(data['Power_hybrid'])
            energy_hybrid = power_hybrid * t_diff / 3600 / 1000
            energy_hybrid_cumulative = energy_hybrid.cumsum()

        # 그래프 그리기
        ax.set_xlabel('Time (minutes)', fontsize=10 * scaling)
        ax.set_ylabel('Energy (kWh)', fontsize=10 * scaling)
        ax.plot(t_min, energy_data_cumulative, label='Data', color='tab:blue', alpha=0.6)
        ax.plot(t_min, energy_phys_cumulative, label='Physics Model', color='tab:red', alpha=0.6)
        if 'Power_hybrid' in data.columns:
            ax.plot(t_min, energy_hybrid_cumulative, label='Hybrid Model', color='tab:green', alpha=0.6)

        ax.legend(loc='upper left', bbox_to_anchor=(0, 0.99))
        ax.set_title(title, pad=10)
        ax.tick_params(axis='both', which='major', labelsize=10 * scaling)
        ax.text(-0.1, 1.05, marker, transform=ax.transAxes, size=16 * scaling, weight='bold', ha='left')  # 마커 추가

    # 각 사이클에 대한 그래프 그리기
    process_and_plot_power(city_cycle1, axs[0, 0], 'A', 'City Cycle 1')
    process_and_plot_energy(city_cycle1, axs[1, 0], 'B', 'City Cycle 1')
    process_and_plot_power(city_cycle2, axs[0, 1], 'C', 'City Cycle 2')
    process_and_plot_energy(city_cycle2, axs[1, 1], 'D', 'City Cycle 2')
    process_and_plot_power(highway_cycle1, axs[0, 2], 'E', 'Highway Cycle 1')
    process_and_plot_energy(highway_cycle1, axs[1, 2], 'F', 'Highway Cycle 1')
    process_and_plot_power(highway_cycle2, axs[0, 3], 'G', 'Highway Cycle 2')
    process_and_plot_energy(highway_cycle2, axs[1, 3], 'H', 'Highway Cycle 2')

    # 레이아웃 조정 및 그림 저장
    plt.tight_layout()
    save_filename = f'figureS4_{fig_number}.png'
    save_path = os.path.join(fig_save_path, save_filename)
    plt.savefig(save_path, dpi=300)
    plt.close()  # 메모리 해제를 위해 그림 닫기


# 시내 주행 사이클과 고속도로 주행 사이클 CSV 파일이 있는 디렉토리
city_directory = r"D:\SamsungSTF\Data\Cycle\Supplementary\GS\City"
highway_directory = r"D:\SamsungSTF\Data\Cycle\Supplementary\GS\HW"

# 디렉토리에서 파일 읽기
city_files = [os.path.join(city_directory, f) for f in os.listdir(city_directory) if f.endswith('.csv')]
highway_files = [os.path.join(highway_directory, f) for f in os.listdir(highway_directory) if f.endswith('.csv')]

# 파일 리스트 정렬 (일관된 그룹화를 위해)
city_files.sort()
highway_files.sort()

print(city_files)
print(highway_files)

# 파일을 2개씩 묶음으로 그룹화
city_groups = [city_files[i:i + 2] for i in range(0, len(city_files), 2)]
highway_groups = [highway_files[i:i + 2] for i in range(0, len(highway_files), 2)]

# 그룹의 수를 동일하게 맞춤
num_groups = min(len(city_groups), len(highway_groups))

# 각 그룹에 대해 figure4 함수 호출
for i in range(num_groups):
    city_group = city_groups[i]
    highway_group = highway_groups[i]

    city_cycle1 = city_group[0]
    city_cycle2 = city_group[1] if len(city_group) > 1 else city_group[0]  # 파일이 하나만 남았을 경우 동일한 파일로 대체

    highway_cycle1 = highway_group[0]
    highway_cycle2 = highway_group[1] if len(highway_group) > 1 else highway_group[0]

    figure4(city_cycle1, highway_cycle1, city_cycle2, highway_cycle2, i + 1)
