import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
from io import BytesIO
from PIL import Image

# .env 파일에서 API 키 로드 (필요 없으시면 주석 처리하거나 삭제해주세요)
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

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

# 파일 쌍의 수
num_files = min(len(city_files), len(hw_files))

# 축척 표시를 위한 'nice' 반올림 함수 (1, 2, 5, 10 배수로)
def nice_round(x):
    """축척선 길이를 1, 2, 5, 10 배수로 '예쁘게' 반올림."""
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

for i in range(num_files):
    # 한 번의 루프마다 city-CSV와 highway-CSV 두 개를 처리
    csv_files = [city_files[i], hw_files[i]]
    save_path = os.path.join(save_dir, f"figureS9_{i+1}.png")

    # 2x4 Figure 생성
    fig = plt.figure(figsize=(24, 10))
    axes = []
    for j in range(8):
        ax = fig.add_subplot(2, 4, j + 1)
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

        # 좌표 데이터 샘플링 (10초 단위)
        df_sampled = df.resample('10s').first().dropna(subset=['latitude', 'longitude'])

        # 서브플롯 시작 인덱스 (idx=0 → A,B,C,D / idx=1 → E,F,G,H)
        subplot_start = idx * 4

        # ------------------ (1) 지도 (A, E) ------------------ #
        # City(CSV idx=0)라면 A, Highway(CSV idx=1)라면 E
        ax_map = axes[subplot_start]  # A 또는 E에 해당

        # 샘플링된 위도/경도
        latitudes = df_sampled['latitude'].tolist()
        longitudes = df_sampled['longitude'].tolist()

        if latitudes and longitudes:
            center_lat = np.mean(latitudes)
            center_lon = np.mean(longitudes)

            # Folium이 아니라, Google Static Map으로 지도 생성
            # ------------------ 줌 레벨 설정 ------------------ #
            # (원래 Folium 코드에서 i, idx에 따라 줌을 달리했으므로 아래 반영)
            if idx == 0:  # City
                if i == 0:
                    zoom = 13
                elif i == 1:
                    zoom = 14
                elif i == 2:
                    zoom = 14
                elif i == 3:
                    zoom = 13
                else:
                    zoom = 14
            else:         # Highway
                if i == 0:
                    zoom = 10
                elif i == 1:
                    zoom = 11
                elif i == 2:
                    zoom = 10
                elif i == 3:
                    zoom = 11
                else:
                    zoom = 12
            # ------------------------------------------------- #

            # 지도 이미지 크기
            size = "800x1000"  # 가로 x 세로 (픽셀)

            # 경로 (Polyline) 파라미터 생성 (빨간색: 0xff0000, 굵기: 5)
            path_coords = "|".join(f"{lat},{lon}" for lat, lon in zip(latitudes, longitudes))
            path_param = f"color:0xff0000|weight:5|{path_coords}"

            # Static Maps API 요청 URL
            base_url = "https://maps.googleapis.com/maps/api/staticmap"
            params = {
                "center": f"{center_lat},{center_lon}",
                "zoom": str(zoom),
                "size": size,
                "maptype": "roadmap",
                "path": path_param,
                "language": "en",        # 영어 라벨
                "key": GOOGLE_API_KEY
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                # 응답받은 지도 이미지를 열기
                img = Image.open(BytesIO(response.content))

                # 구글 지도 하단 로고 등을 어느 정도 크롭해서 제거 (5%)
                width, height = img.size
                crop_ratio = 0.05  # 하단 5% 정도만 크롭
                new_height = int(height * (1 - crop_ratio))
                img = img.crop((0, 0, width, new_height))

                # Axes에 이미지 표시
                ax_map.imshow(img, origin='upper')
                ax_map.axis('off')

                # ----- 축척(Scale bar) 계산 후 지도 위에 그리기 -----
                # 해상도 (m/픽셀)
                resolution = 156543.03392 * np.cos(np.deg2rad(center_lat)) / (2 ** zoom)

                # 축척길이를 지도 너비의 20%로 설정해보고 실제 미터값으로 변환
                bar_px_initial = 0.2 * width
                raw_length_m = bar_px_initial * resolution
                # 1,2,5,10 단위로 '예쁘게' 반올림
                bar_length_m = nice_round(raw_length_m)
                # 해당 미터값을 다시 픽셀로 환산
                bar_px = bar_length_m / resolution

                # 축척선 시작 위치 (좌측 하단)
                x_start = 10
                y_pos = new_height - 10  # 하단에서 10px 위

                # 축척선 그리기
                ax_map.plot([x_start, x_start + bar_px], [y_pos, y_pos], color='black', lw=4)

                # 축척 문구: km 혹은 m 단위
                if bar_length_m >= 1000:
                    scale_text = f"{bar_length_m / 1000:.1f} km"
                else:
                    scale_text = f"{bar_length_m:.0f} m"

                # 축척 글자
                ax_map.text(x_start, y_pos - 10, scale_text,
                            color='black', fontsize=16 * scaling, fontweight='bold',
                            verticalalignment='bottom')

                # 서브플롯 제목
                if idx == 0:
                    ax_map.set_title('City Cycle Route Map')
                else:
                    ax_map.set_title('Highway Cycle Route Map')

                # 서브플롯 레이블 (A 또는 E)
                label_text = labels[subplot_start]
                fontsize_ = 16 * scaling
                ax_map.text(-0.1, 1.05, label_text, transform=ax_map.transAxes,
                            fontsize=fontsize_, fontweight='bold', va='bottom', ha='right')
            else:
                # 지도 요청 실패 시
                ax_map.text(0.5, 0.5, 'Failed to load map', ha='center', va='center')
                ax_map.axis('off')
        else:
            ax_map.text(0.5, 0.5, 'No sampled coordinates', ha='center', va='center')
            ax_map.set_axis_off()

        # ------------------ (2) 속도/가속도 (B, F) ------------------ #
        ax_speed = axes[subplot_start + 1]
        ax_accel = ax_speed.twinx()

        # 속도: km/h
        speed_kmh = df['speed'] * 3.6
        ax_speed.plot(df['elapsed_time_min'], speed_kmh, color="#0073C2FF", label='Speed')
        ax_accel.plot(df['elapsed_time_min'], df['acceleration'], color="#EFC000FF", label='Acceleration')

        ax_speed.set_xlabel('Time (minutes)')
        ax_speed.set_ylabel('Speed (km/h)', color="#0073C2FF")
        ax_accel.set_ylabel('Acceleration (m/s²)', color="#EFC000FF")

        ax_speed.tick_params(axis='y', labelcolor="#0073C2FF")
        ax_accel.tick_params(axis='y', labelcolor="#EFC000FF")

        ax_speed.set_title('Speed and Acceleration')

        # 범례
        lines_1, labels_1 = ax_speed.get_legend_handles_labels()
        lines_2, labels_2 = ax_accel.get_legend_handles_labels()
        ax_speed.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        label_text = labels[subplot_start + 1]
        fontsize_ = 16 * scaling
        ax_speed.text(-0.1, 1.05, label_text, transform=ax_speed.transAxes,
                      fontsize=fontsize_, fontweight='bold', va='bottom', ha='right')

        # ------------------ (3) 파워 (C, G) ------------------ #
        ax_power = axes[subplot_start + 2]
        ax_power.plot(df['elapsed_time_min'], df['Power_phys'] / 1000, color="#CD534CFF",
                      label='Physics Model', alpha=0.7)
        ax_power.plot(df['elapsed_time_min'], df['Power_hybrid'] / 1000, color="#20854EFF",
                      label='Hybrid Model(XGB)', alpha=0.7)

        ax_power.set_xlabel('Time (minutes)')
        ax_power.set_ylabel('Power (kW)')
        ax_power.set_title('Power')
        ax_power.legend(loc='upper left')

        label_text = labels[subplot_start + 2]
        fontsize_ = 16 * scaling
        ax_power.text(-0.1, 1.05, label_text, transform=ax_power.transAxes,
                      fontsize=fontsize_, fontweight='bold', va='bottom', ha='right')

        # ------------------ (4) 누적 에너지 (D, H) ------------------ #
        ax_energy = axes[subplot_start + 3]

        # 누적 에너지 계산 (kWh)
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

        energy_phys = np.array(energy_phys[:len(df)])
        energy_hybrid = np.array(energy_hybrid[:len(df)])

        ax_energy.plot(df['elapsed_time_min'], energy_phys, color="#CD534CFF",
                       label='Physics Model', alpha=0.7)
        ax_energy.plot(df['elapsed_time_min'], energy_hybrid, color="#20854EFF",
                       label='Hybrid Model(XGB)', alpha=0.7)

        ax_energy.set_xlabel('Time (minutes)')
        ax_energy.set_ylabel('Energy (kWh)')
        ax_energy.set_title('Cumulative Energy')
        ax_energy.legend(loc='upper left')

        label_text = labels[subplot_start + 3]
        fontsize_ = 16 * scaling
        ax_energy.text(-0.1, 1.05, label_text, transform=ax_energy.transAxes,
                       fontsize=fontsize_, fontweight='bold', va='bottom', ha='right')

    # 남는 서브플롯이 있다면 숨기기
    total_plots = len(csv_files) * 4
    for j in range(total_plots, len(axes)):
        axes[j].axis('off')

    # 레이아웃 및 저장
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved at {save_path}")
