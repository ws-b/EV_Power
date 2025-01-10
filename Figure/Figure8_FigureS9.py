# 필요한 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
from io import BytesIO
from PIL import Image

# WebDriverWait을 위한 라이브러리 추가
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 디렉토리 경로
city_dir = r"D:\SamsungSTF\Data\Cycle\Supplementary\KOTI\City"
hw_dir = r"D:\SamsungSTF\Data\Cycle\Supplementary\KOTI\HW"

# CSV 파일 목록 가져오기
city_files = sorted([os.path.join(city_dir, f) for f in os.listdir(city_dir) if f.endswith('.csv')])
hw_files = sorted([os.path.join(hw_dir, f) for f in os.listdir(hw_dir) if f.endswith('.csv')])

# 저장 경로
save_dir = r"C:\Users\BSL\Desktop\Figures\Supplementary"

# 서브플롯 레이블
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

scaling = 1.4
# Set font sizes using the scaling factor
plt.rcParams['font.size'] = 10 * scaling  # Base font size
plt.rcParams['axes.titlesize'] = 12 * scaling  # Title font size
plt.rcParams['axes.labelsize'] = 10 * scaling  # Axis label font size
plt.rcParams['xtick.labelsize'] = 10 * scaling  # X-axis tick label font size
plt.rcParams['ytick.labelsize'] = 10 * scaling  # Y-axis tick label font size
plt.rcParams['legend.fontsize'] = 10 * scaling  # Legend font size
plt.rcParams['legend.title_fontsize'] = 10 * scaling  # Legend title font size
plt.rcParams['figure.titlesize'] = 12 * scaling  # Figure title font size

# 파일 쌍의 수 (4개)
num_files = min(len(city_files), len(hw_files))

for i in range(num_files):
    csv_files = [city_files[i], hw_files[i]]
    save_path = os.path.join(save_dir, f"figureS9_{i+1}.png")

    # 2x4 Figure 생성
    fig = plt.figure(figsize=(24, 10))

    axes = []
    for j in range(8):
        ax = fig.add_subplot(2, 4, j+1)
        axes.append(ax)

    for idx, csv_file in enumerate(csv_files):
        # CSV 파일 읽기
        df = pd.read_csv(csv_file)

        # 시간 데이터 처리
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        start_time = df['time'].iloc[0]
        df['elapsed_time_sec'] = (df['time'] - start_time).dt.total_seconds()
        df['elapsed_time_min'] = df['elapsed_time_sec'] / 60
        df.set_index('time', inplace=True)

        # 좌표 데이터 샘플링
        df_sampled = df.resample('10s').first().dropna(subset=['latitude', 'longitude'])

        # 서브플롯 시작 인덱스
        subplot_start = idx * 4

        ### 첫 번째 플롯: Folium 지도로 경로 표시 ###
        ax_map = axes[subplot_start]

        # 샘플링된 위도와 경도 추출
        latitudes = df_sampled['latitude'].tolist()
        longitudes = df_sampled['longitude'].tolist()

        if latitudes and longitudes:
            # 중심 좌표 계산
            center_lat = np.mean(latitudes)
            center_lon = np.mean(longitudes)

            # Folium 지도 생성 시 타일 옵션 설정 (컬러 지도와 영어 라벨)
            tile_option = 'CartoDB positron'  # 컬러 지도와 영어 라벨

            # 줌 레벨 설정 및 줌 컨트롤 제거, 축척 표시 추가
            if i == 0:
                zoom_city = 14
                zoom_hw = 11
            elif i == 1:
                zoom_city = 15
                zoom_hw = 12
            elif i == 2:
                zoom_city = 15
                zoom_hw = 11
            elif i == 3:
                zoom_city = 14
                zoom_hw = 12
            else:
                zoom_city = 15
                zoom_hw = 12
            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_city if idx == 0 else zoom_hw,
                           tiles=tile_option, zoom_control=False, control_scale=True)

            # 축척 위치를 왼쪽 하단으로 이동시키고 크기를 키우는 CSS 추가
            scale_position_css = '''
            <style>
            .leaflet-control-scale {
                position: absolute !important;
                bottom: 10px !important;
                left: 10px !important;
                right: auto !important;
            }
            </style>
            '''
            m.get_root().html.add_child(folium.Element(scale_position_css))

            # 경로 추가
            folium.PolyLine(list(zip(latitudes, longitudes)), color="blue", weight=5).add_to(m)

            # 지도를 HTML로 저장
            map_html = f"map_{i}_{idx}.html"
            m.save(map_html)

            # Selenium을 사용하여 지도 이미지를 캡처
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--window-size=1600x1600")
            options.add_argument("--hide-scrollbars")
            driver = webdriver.Chrome(options=options)
            driver.get(f"file://{os.getcwd()}/{map_html}")

            # 축척이 로드될 때까지 기다리기 (최대 10초)
            try:
                wait = WebDriverWait(driver, 10)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "leaflet-control-scale-line")))
                time.sleep(1)  # 추가로 1초 대기하여 완전히 로드되도록 함
            except Exception as e:
                print(f"축척을 찾을 수 없습니다: {e}")

            # 지도의 스크린샷 캡처
            png = driver.get_screenshot_as_png()
            driver.quit()

            # 이미지를 읽어와서 표시
            img = Image.open(BytesIO(png))

            # 이미지를 Axes에 표시 (origin='upper'로 설정하여 y축 방향 수정)
            ax_map.imshow(img, origin='upper')
            ax_map.axis('off')

            # 우측 상단에 'N' 표시
            ax_map.text(0.95, 0.95, 'N', transform=ax_map.transAxes, fontsize=16, fontweight='bold',
                        va='top', ha='right',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.1'))

            # 제목 설정
            if idx == 0:
                ax_map.set_title('City Cycle Route Map')
            else:
                ax_map.set_title('Highway Cycle Route Map')

            # 서브플롯 레이블 추가
            label_text = labels[subplot_start]
            fontsize = 16 * scaling
            ax_map.text(-0.1, 1.05, label_text, transform=ax_map.transAxes, fontsize=fontsize, fontweight='bold',
                        va='bottom', ha='right')
        else:
            ax_map.text(0.5, 0.5, 'No sampled coordinates', horizontalalignment='center', verticalalignment='center')
            ax_map.set_axis_off()

        ### 두 번째 플롯: 속도와 가속도 ###
        ax_speed = axes[subplot_start + 1]
        ax_accel = ax_speed.twinx()

        # 속도: km/h로 변환
        speed_kmh = df['speed'] * 3.6
        ax_speed.plot(df['elapsed_time_min'], speed_kmh, color="#0073C2FF", label='Speed')

        # 가속도
        ax_accel.plot(df['elapsed_time_min'], df['acceleration'], color="#EFC000FF", label='Acceleration')

        ax_speed.set_xlabel('Time (minutes)')
        ax_speed.set_ylabel('Speed (km/h)', color="#0073C2FF")
        ax_accel.set_ylabel('Acceleration (m/s²)', color="#EFC000FF")

        ax_speed.tick_params(axis='y', labelcolor="#0073C2FF")
        ax_accel.tick_params(axis='y', labelcolor="#EFC000FF")

        ax_speed.set_title('Speed and Acceleration')

        # 범례 추가
        lines_1, labels_1 = ax_speed.get_legend_handles_labels()
        lines_2, labels_2 = ax_accel.get_legend_handles_labels()
        ax_speed.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        # 서브플롯 레이블 추가
        label_text = labels[subplot_start + 1]
        fontsize = 16 * scaling
        ax_speed.text(-0.1, 1.05, label_text, transform=ax_speed.transAxes, fontsize=fontsize, fontweight='bold',
                      va='bottom', ha='right')

        ### 세 번째 플롯: 파워 하이브리드와 파워 피지컬 ###
        ax_power = axes[subplot_start + 2]

        # 파워 플롯
        ax_power.plot(df['elapsed_time_min'], df['Power_phys']/1000, color="#CD534CFF", label='Physics Model', alpha=0.7)
        ax_power.plot(df['elapsed_time_min'], df['Power_hybrid'] / 1000, color="#20854EFF", label='Hybrid Model(XGB)', alpha=0.7)

        ax_power.set_xlabel('Time (minutes)')
        ax_power.set_ylabel('Power (kW)')
        ax_power.set_title('Power')

        ax_power.legend(loc='upper left')

        # 서브플롯 레이블 추가
        label_text = labels[subplot_start + 2]
        fontsize = 16 * scaling
        ax_power.text(-0.1, 1.05, label_text, transform=ax_power.transAxes, fontsize=fontsize, fontweight='bold',
                      va='bottom', ha='right')

        ### 네 번째 플롯: 누적 에너지 ###
        ax_energy = axes[subplot_start + 3]

        # 누적 에너지 계산
        elapsed_time_hours = df['elapsed_time_min'] / 60

        energy_phys = [0]
        energy_hybrid = [0]

        for k in range(1, len(df)):
            t_prev = elapsed_time_hours.iloc[k-1]
            t_curr = elapsed_time_hours.iloc[k]
            dt = t_curr - t_prev

            p_phys_avg = (df['Power_phys'].iloc[k-1] + df['Power_phys'].iloc[k]) / 2 / 1000  # kW
            p_hybrid_avg = (df['Power_hybrid'].iloc[k-1] + df['Power_hybrid'].iloc[k]) / 2 / 1000  # kW

            energy_phys.append(energy_phys[-1] + p_phys_avg * dt)
            energy_hybrid.append(energy_hybrid[-1] + p_hybrid_avg * dt)

        energy_phys = np.array(energy_phys)
        energy_hybrid = np.array(energy_hybrid)

        energy_phys = energy_phys[:len(df)]
        energy_hybrid = energy_hybrid[:len(df)]

        # 누적 에너지 플롯
        ax_energy.plot(df['elapsed_time_min'], energy_phys, color="#CD534CFF", label='Physics Model', alpha=0.7)
        ax_energy.plot(df['elapsed_time_min'], energy_hybrid, color="#20854EFF", label='Hybrid Model(XGB)', alpha=0.7)

        ax_energy.set_xlabel('Time (minutes)')
        ax_energy.set_ylabel('Energy (kWh)')
        ax_energy.set_title('Cumulative Energy')

        ax_energy.legend(loc='upper left')

        # 서브플롯 레이블 추가
        label_text = labels[subplot_start + 3]
        fontsize = 16 * scaling
        ax_energy.text(-0.1, 1.05, label_text, transform=ax_energy.transAxes, fontsize=fontsize, fontweight='bold',
                      va='bottom', ha='right')

    # 남은 서브플롯 숨기기 (만약 필요하다면)
    total_plots = len(csv_files) * 4
    for j in range(total_plots, len(axes)):
        axes[j].axis('off')

    # 레이아웃 조정 및 저장
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved at {save_path}")