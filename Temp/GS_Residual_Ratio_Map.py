import os
import pandas as pd
import matplotlib.pyplot as plt
import requests
from PIL import Image
from datetime import datetime
import glob
from tqdm import tqdm
from GS_vehicle_dict import vehicle_dict
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

# Google Maps API Key
api_key = 'AIzaSyAqsoklc9VL03MArm2H-Fb4LaAzIyi-y2E'

# -------------------- 설정 --------------------

# CSV 파일들이 있는 디렉토리 경로를 설정합니다.
directory = r'D:\SamsungSTF\Processed_Data\TripByTrip'
selected_car = 'Ioniq5'

# 그래프를 저장할 PDF 파일의 경로를 설정합니다.
output_pdf = rf'D:\SamsungSTF\Processed_Data\Graphs\all_graphs_{selected_car}.pdf'

# 저장 디렉토리가 존재하지 않으면 생성합니다.
os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

device_ids = vehicle_dict.get(selected_car, [])

if not device_ids:
    print(f"No device IDs found for the selected vehicle: {selected_car}")
    exit()

# 디렉토리에서 'bms'와 'altitude'가 포함된 모든 CSV 파일을 가져옵니다.
all_files = glob.glob(os.path.join(directory, '*bms*altitude*09*.csv'))

# 단말기 번호가 파일명에 포함된 파일만 선택합니다.
files = [file for file in all_files if any(device_id in os.path.basename(file) for device_id in device_ids)]

if not files:
    print(f"No files found for the selected vehicle: {selected_car}")
    exit()

print(f"Processing {len(files)} files for vehicle: {selected_car}")

# -------------------- 함수 정의 --------------------

def plot_multi_yaxes_shared(df, x_col, y_cols_grouped, labels_grouped, colors_grouped, title, alpha=0.7):
    """
    멀티 y축 그래프를 생성하는 함수. y_cols_grouped는 각 y축에 공유할 y_col들의 리스트입니다.

    Parameters:
        df (DataFrame): 데이터프레임.
        x_col (str): x축으로 사용할 컬럼명.
        y_cols_grouped (list of lists): 각 y축에 공유할 y_col들의 리스트.
        labels_grouped (list of lists): 각 y축의 레이블 리스트.
        colors_grouped (list of lists): 각 y축의 색상 리스트.
        title (str): 그래프 제목.
        alpha (float): 선의 투명도.

    Returns:
        fig (Figure): 생성된 matplotlib Figure 객체.
    """
    fig, ax_main = plt.subplots(figsize=(14, 8))
    ax_main.set_xlabel('Elapsed Time (s)')  # Updated x-axis label

    axes = [ax_main]
    lines = []
    labels_legend = []

    for i, (y_cols, labels, colors) in enumerate(zip(y_cols_grouped, labels_grouped, colors_grouped)):
        if i == 0:
            ax = ax_main
        else:
            ax = ax_main.twinx()
            # y축 간격을 조정하여 겹치지 않도록 함
            ax.spines['right'].set_position(('outward', 60 * (i - 1)))
            axes.append(ax)

        for y_col, label, color in zip(y_cols, labels, colors):
            ax.plot(df[x_col], df[y_col], color=color, label=label, alpha=alpha)
            ax.set_ylabel(labels[0], color=colors[0])  # Use the first label and color for the axis
            ax.tick_params(axis='y', labelcolor=colors[0])

            # Collect legend handles
            line = ax.get_lines()[-1]  # Get the last line
            lines.append(line)
            labels_legend.append(label)

    # 레전드 추가
    plt.legend(lines, labels_legend, loc='upper left')

    plt.title(title)
    fig.tight_layout()
    return fig

def cumulative_trapz(y, x):
    """
    Numpy의 trapz를 사용하여 누적 트라페조이달 적분을 수동으로 구현한 함수.

    Parameters:
        y (array-like): y 데이터.
        x (array-like): x 데이터.

    Returns:
        cumulative_integral (list): 누적 적분 결과 리스트.
    """
    cumulative_integral = [0.0]
    for i in range(1, len(y)):
        integral = cumulative_integral[-1] + 0.5 * (y[i - 1] + y[i]) * (x[i] - x[i - 1])
        cumulative_integral.append(integral)
    return cumulative_integral


def get_google_map_image(df, api_key, map_file):
    """
    Google Static Maps API를 사용하여 GPS 경로를 포함한 지도를 PNG로 저장하는 함수.

    Parameters:
        df (DataFrame): GPS 데이터가 포함된 데이터프레임 (lat, lng 포함).
        api_key (str): Google Maps API Key.
        map_file (str): 저장할 이미지 파일 경로.

    Returns:
        image_file (str): 저장된 이미지 파일 경로.
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"

    # 좌표 경로를 문자열로 생성
    path = "|".join([f"{row['lat']},{row['lng']}" for _, row in df.iterrows()])

    # 요청 URL 구성
    params = {
        'size': '640x640',  # 지도 크기
        'path': f"color:0x0000ff|weight:5|{path}",  # 경로 설정 (파란색 선)
        'key': api_key,
        'maptype': 'roadmap'
    }

    # Google Static Maps API에 요청
    response = requests.get(base_url, params=params)

    # 응답이 성공적이면 이미지를 저장
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save(map_file)
        return map_file
    else:
        print(f"Error fetching map: {response.status_code}")
        return None

def process_and_save_graphs(file, pdf, selected_car):
    """
    특정 CSV 파일을 처리하고, 두 개의 그래프과 GPS 그래프를 PDF에 저장하는 함수.

    Parameters:
        file (str): 처리할 CSV 파일 경로.
        pdf (PdfPages): PdfPages 객체.
        selected_car (str): 선택한 차량명.
    """
    df = pd.read_csv(file)

    # 'time' 컬럼을 datetime 형식으로 변환
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

    # 'Residual_Ratio' 컬럼 계산
    df['Residual_Ratio'] = (df['Power_data'] - df['Power_phys']) / df['Power_phys']

    # 'altitude'의 유니크 값 수 확인
    if df['altitude'].nunique() <= 3:
        print(f"Skipping file {file} due to <=3 unique altitude values.")
        return

    # 시간순으로 정렬
    df.sort_values('time', inplace=True)

    # 'elapsed_time' 컬럼 추가: 시작점 0
    df['elapsed_time'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

    # 'altitude' 선형 보간 (1분 간격 -> 2초 간격)
    df['altitude'] = df['altitude'].interpolate(method='linear')

    # 트라페조이달 적분 계산
    # 'elapsed_time'을 초 단위로 사용
    elapsed_time = df['elapsed_time'].values

    # 'Power_phys'와 'Power_data'의 누적 트라페조이달 적분 (Numpy trapz 사용)
    df['Power_phys_integral'] = cumulative_trapz(df['Power_phys'].values, elapsed_time)
    df['Power_data_integral'] = cumulative_trapz(df['Power_data'].values, elapsed_time)

    # -------------------- 데이터 단위 변환 --------------------
    # Power 데이터를 kW 단위로 변환
    df['Power_phys_kW'] = df['Power_phys'] / 1000.0
    df['Power_data_kW'] = df['Power_data'] / 1000.0

    # 적분된 Power 데이터를 kWh 단위로 변환
    df['Energy_phys'] = df['Power_phys_integral'] / 3600000.0  # W*s to kWh
    df['Energy_data'] = df['Power_data_integral'] / 3600000.0  # W*s to kWh

    # -------------------- 첫 번째 그래프: 멀티 y축 --------------------
    # y_cols_grouped: 각 그룹이 공유할 y_cols 리스트
    y_cols_grouped_1 = [
        ['Power_phys_kW', 'Power_data_kW'],  # Shared Y-axis
        ['altitude'],
        ['Residual_Ratio']
    ]

    labels_grouped_1 = [
        ['Power_phys (kW)', 'Power_data (kW)'],
        ['Altitude'],
        ['Residual_Ratio']
    ]

    colors_grouped_1 = [
        ['tab:red', 'tab:blue'],
        ['tab:green'],
        ['tab:purple']
    ]

    fig1 = plot_multi_yaxes_shared(
        df=df,
        x_col='elapsed_time',
        y_cols_grouped=y_cols_grouped_1,
        labels_grouped=labels_grouped_1,
        colors_grouped=colors_grouped_1,
        title=f"{selected_car} - Power, Altitude & Residual_Ratio over Time\nFile: {os.path.basename(file)}",
        alpha=0.7  # 투명도 설정
    )
    pdf.savefig(fig1)
    plt.close(fig1)

    # -------------------- 두 번째 그래프: 적분 값 및 멀티 y축 --------------------
    y_cols_grouped_2 = [
        ['Energy_phys', 'Energy_data'],  # Shared Y-axis
        ['altitude'],
        ['Residual_Ratio']
    ]

    labels_grouped_2 = [
        ['Energy_phys (kWh)', 'Energy_data (kWh)'],
        ['Altitude'],
        ['Residual_Ratio']
    ]

    colors_grouped_2 = [
        ['tab:red', 'tab:blue'],
        ['tab:green'],
        ['tab:purple']
    ]

    fig2 = plot_multi_yaxes_shared(
        df=df,
        x_col='elapsed_time',
        y_cols_grouped=y_cols_grouped_2,
        labels_grouped=labels_grouped_2,
        colors_grouped=colors_grouped_2,
        title=f"{selected_car} - Integrated Energy, Altitude & Residual_Ratio over Time\nFile: {os.path.basename(file)}",
        alpha=0.7  # 투명도 설정
    )
    pdf.savefig(fig2)
    plt.close(fig2)

    # -------------------- 세 번째 그래프: GPS 정보 --------------------
    # GPS 좌표가 있는지 확인
    if 'lat' in df.columns and 'lng' in df.columns:
        gps_df = df[['lat', 'lng']].dropna()
        if not gps_df.empty:
            image_file = f"gps_map_{os.path.basename(file).replace('.csv', '.png')}"
            png_file = get_google_map_image(gps_df, api_key, image_file)

            if png_file:
                # PNG 파일을 PDF에 추가
                img = Image.open(png_file)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig)
                plt.close(fig)
        else:
            print(f"No GPS data to plot in file {file}.")
    else:
        print(f"'lat' and/or 'lng' columns not found in file {file}.")
# -------------------- 메인 처리 루프 --------------------

with PdfPages(output_pdf) as pdf:
    for file in tqdm(files[:40], desc="Processing Files"):
        try:
            process_and_save_graphs(file, pdf, selected_car)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

print(f"All graphs have been saved to {output_pdf}")
