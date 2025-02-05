import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv  # .env 파일에서 API 키 로드

# CSV 파일 목록
csv_files = [
    r"D:\SamsungSTF\Data\Cycle\City_KOTI\20190101_240493.csv",
    r"D:\SamsungSTF\Data\Cycle\HW_KOTI\20190119_1903235.csv"
]

# 저장 경로
save_path = r"C:\Users\BSL\Desktop\Figures\figure8.png"

# 서브플롯 레이블
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# 2x4 Figure 생성
fig = plt.figure(figsize=(24, 10))

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

axes = []
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1)
    axes.append(ax)

# API 키 로드 (.env 파일 내 GOOGLE_API_KEY 변수)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")


# 축척 표시를 위한 'nice' 반올림 함수 (1, 2, 5, 10 배수로)
def nice_round(x):
    if x <= 0:
        return x
    exponent = np.floor(np.log10(x))
    fraction = x / (10 ** exponent)
    if fraction < 1.5:
        nice = 1
    elif fraction < 3:
        nice = 2
    elif fraction < 7:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** exponent)


# 각 CSV 파일 처리
for idx, csv_file in enumerate(csv_files):
    # CSV 파일 읽기
    df = pd.read_csv(csv_file)

    # 시간 데이터 처리
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    start_time = df['time'].iloc[0]
    df['elapsed_time_sec'] = (df['time'] - start_time).dt.total_seconds()
    df['elapsed_time_min'] = df['elapsed_time_sec'] / 60
    df.set_index('time', inplace=True)

    # 좌표 데이터 샘플링 (10초 단위)
    df_sampled = df.resample('10s').first().dropna(subset=['latitude', 'longitude'])

    # 서브플롯 시작 인덱스 (각 CSV당 4개의 플롯)
    subplot_start = idx * 4

    ### 첫 번째 플롯: Google Static Maps API를 이용한 지도 ###
    ax_map = axes[subplot_start]

    # 샘플링된 위도와 경도 추출
    latitudes = df_sampled['latitude'].tolist()
    longitudes = df_sampled['longitude'].tolist()

    if latitudes and longitudes:
        # 중심 좌표 계산
        center_lat = np.mean(latitudes)
        center_lon = np.mean(longitudes)
        zoom = 13 if idx == 0 else 11

        # 지도 이미지 사이즈 (width x height, 픽셀 단위)
        size = "800x1000"

        # 경로(Polyline) 파라미터 생성 (빨간색: 0xff0000)
        path_coords = "|".join(f"{lat},{lon}" for lat, lon in zip(latitudes, longitudes))
        path_param = f"color:0xff0000|weight:5|{path_coords}"

        # Static Maps API 요청 URL 구성 (언어는 영어로 지정)
        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{center_lat},{center_lon}",
            "zoom": str(zoom),
            "size": size,
            "maptype": "roadmap",
            "path": path_param,
            "language": "en",
            "key": GOOGLE_API_KEY
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            # 응답받은 이미지 열기
            img = Image.open(BytesIO(response.content))

            # 하단 크롭 비율을 10%에서 5%로 줄여서 필요한 정보(예: 축척)가 남도록 함
            width, height = img.size
            crop_ratio = 0.05  # 하단 5%만 크롭
            new_height = int(height * (1 - crop_ratio))
            img = img.crop((0, 0, width, new_height))

            ax_map.imshow(img, origin='upper')
            ax_map.axis('off')

            # (기존 'N' 표시 코드는 제거)

            # ----- 여기서 축척(scale bar) 추가 -----
            # 구글 지도 해상도 (미터/픽셀): 156543.03392 * cos(latitude) / (2^zoom)
            resolution = 156543.03392 * np.cos(np.deg2rad(center_lat)) / (2 ** zoom)
            # 이미지의 좌표는 픽셀 단위 (x: 0~width, y: 0~new_height)
            # 축척 길이를 이미지 너비의 20% 정도로 잡음
            bar_px_initial = 0.2 * width
            raw_length_m = bar_px_initial * resolution
            # 깔끔한 숫자로 반올림 (예, 1, 2, 5, 10의 배수)
            bar_length_m = nice_round(raw_length_m)
            # 반올림한 실제 거리값에 해당하는 픽셀 길이
            bar_px = bar_length_m / resolution

            # 축척선은 이미지의 왼쪽 아래 (여백 10px)
            x_start = 10
            y_pos = new_height - 10  # 하단에서 10px 위

            # 축척선 그리기 (굵은 검은 선)
            ax_map.plot([x_start, x_start + bar_px], [y_pos, y_pos], color='black', lw=4)

            # 축척 텍스트: 1000m 이상이면 km 단위로 표시
            if bar_length_m >= 1000:
                scale_text = f"{bar_length_m / 1000:.1f} km"
            else:
                scale_text = f"{bar_length_m:.0f} m"
            # 텍스트는 축척선 위쪽에 표시
            ax_map.text(x_start, y_pos - 10, scale_text, color='black', fontsize=16, fontweight='bold',
                        verticalalignment='bottom')
            # -----------------------------------------

            # 제목 설정
            if idx == 0:
                ax_map.set_title('City Cycle Route Map')
            else:
                ax_map.set_title('Highway Cycle Route Map')

            # 서브플롯 레이블 추가 (예: 'A', 'E', ...)
            label_text = labels[subplot_start]
            fontsize = 16 * scaling
            ax_map.text(-0.1, 1.05, label_text, transform=ax_map.transAxes,
                        fontsize=fontsize, fontweight='bold', va='bottom', ha='right')
        else:
            ax_map.text(0.5, 0.5, 'Failed to load map', horizontalalignment='center',
                        verticalalignment='center')
            ax_map.axis('off')
    else:
        ax_map.text(0.5, 0.5, 'No sampled coordinates', horizontalalignment='center', verticalalignment='center')
        ax_map.set_axis_off()

    ### 두 번째 플롯: 속도와 가속도 ###
    ax_speed = axes[subplot_start + 1]
    ax_accel = ax_speed.twinx()

    # 속도: m/s -> km/h 변환
    speed_kmh = df['speed'] * 3.6
    ax_speed.plot(df['elapsed_time_min'], speed_kmh, color="#0073C2FF", label='Speed')
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
    ax_speed.text(-0.1, 1.05, label_text, transform=ax_speed.transAxes,
                  fontsize=fontsize, fontweight='bold', va='bottom', ha='right')

    ### 세 번째 플롯: 파워 (물리 모델 vs 하이브리드 모델) ###
    ax_power = axes[subplot_start + 2]
    ax_power.plot(df['elapsed_time_min'], df['Power_phys'] / 1000, color="#CD534CFF",
                  label='Physics Model', alpha=0.7)
    ax_power.plot(df['elapsed_time_min'], df['Power_hybrid'] / 1000, color="#20854EFF",
                  label='Hybrid Model(XGB)', alpha=0.7)

    ax_power.set_xlabel('Time (minutes)')
    ax_power.set_ylabel('Power (kW)')
    ax_power.set_title('Power')
    ax_power.legend(loc='upper left')

    # 서브플롯 레이블 추가
    label_text = labels[subplot_start + 2]
    fontsize = 16 * scaling
    ax_power.text(-0.1, 1.05, label_text, transform=ax_power.transAxes,
                  fontsize=fontsize, fontweight='bold', va='bottom', ha='right')

    ### 네 번째 플롯: 누적 에너지 ###
    ax_energy = axes[subplot_start + 3]

    # 누적 에너지 계산 (kWh)
    elapsed_time_hours = df['elapsed_time_min'] / 60

    energy_phys = [0]
    energy_hybrid = [0]

    for i in range(1, len(df)):
        t_prev = elapsed_time_hours.iloc[i - 1]
        t_curr = elapsed_time_hours.iloc[i]
        dt = t_curr - t_prev

        p_phys_avg = (df['Power_phys'].iloc[i - 1] + df['Power_phys'].iloc[i]) / 2 / 1000  # kW
        p_hybrid_avg = (df['Power_hybrid'].iloc[i - 1] + df['Power_hybrid'].iloc[i]) / 2 / 1000  # kW

        energy_phys.append(energy_phys[-1] + p_phys_avg * dt)
        energy_hybrid.append(energy_hybrid[-1] + p_hybrid_avg * dt)

    energy_phys = np.array(energy_phys)
    energy_hybrid = np.array(energy_hybrid)
    energy_phys = energy_phys[:len(df)]
    energy_hybrid = energy_hybrid[:len(df)]

    # 누적 에너지 플롯
    ax_energy.plot(df['elapsed_time_min'], energy_phys, color="#CD534CFF",
                   label='Physics Model', alpha=0.7)
    ax_energy.plot(df['elapsed_time_min'], energy_hybrid, color="#20854EFF",
                   label='Hybrid Model(XGB)', alpha=0.7)

    ax_energy.set_xlabel('Time (minutes)')
    ax_energy.set_ylabel('Energy (kWh)')
    ax_energy.set_title('Cumulative Energy')
    ax_energy.legend(loc='upper left')

    # 서브플롯 레이블 추가
    label_text = labels[subplot_start + 3]
    fontsize = 16 * scaling
    ax_energy.text(-0.1, 1.05, label_text, transform=ax_energy.transAxes,
                   fontsize=fontsize, fontweight='bold', va='bottom', ha='right')

# 남은 서브플롯 숨기기
total_plots = len(csv_files) * 4
for i in range(total_plots, len(axes)):
    axes[i].axis('off')

# 레이아웃 조정 및 저장
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
