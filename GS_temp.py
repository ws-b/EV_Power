import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯을 위해 추가
import plotly.express as px  # 인터랙티브 플롯을 위해 추가
from concurrent.futures import ProcessPoolExecutor, as_completed
from GS_vehicle_dict import vehicle_dict


def process_single_file(file):
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            # 작은 epsilon을 추가하여 분모가 0이 되는 것을 방지합니다.
            epsilon = 1e-6
            data['Residual_Ratio'] = data['Residual'] / (data['Power_phys'] + epsilon)
            # 'time' 열을 datetime 형식으로 변환
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            data['speed'] = 3.6 * data['speed']
            # 'jerk' 계산 (가속도의 변화율)
            data['jerk'] = data['acceleration'].diff().fillna(0)

            # 필요한 열만 반환
            return data[
                ['time', 'speed', 'acceleration', 'jerk', 'ext_temp', 'Residual', 'Residual_Ratio', 'Power_phys',
                 'Power_data']]
        else:
            print(f"File {file} does not contain required columns 'Power_phys' and 'Power_data'.")
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None


def process_files(files):
    df_list = []
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                data = future.result()
                if data is not None:
                    # 데이터에 파일명을 추가하여 추적할 수 있도록 합니다.
                    data['source_file'] = os.path.basename(file)
                    df_list.append(data)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    full_data = pd.concat(df_list, ignore_index=True)
    return full_data


def plot_residual_ratio_distribution(data):
    # 데이터에서 NaN 및 무한대 값을 제거합니다.
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Residual_Ratio'])

    # 이상치 제거 (예: Residual_Ratio가 상위 1% 이상인 경우)
    upper_limit = data['Residual_Ratio'].quantile(0.99)
    data = data[data['Residual_Ratio'] <= upper_limit]

    # Residual_Ratio vs Speed
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='speed', y='Residual_Ratio', data=data, alpha=0.5)
    plt.title('Residual Ratio vs. Speed')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Residual Ratio')
    plt.show()

    # Residual_Ratio vs Acceleration
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='acceleration', y='Residual_Ratio', data=data, alpha=0.5)
    plt.title('Residual Ratio vs. Acceleration')
    plt.xlabel('Acceleration (m/s²)')
    plt.ylabel('Residual Ratio')
    plt.show()

def plot_residual_ratio_3d_plotly(data, sample_size=10000):
    # 데이터에서 NaN 및 무한대 값을 제거하고 Residual_Ratio가 0이 아닌 데이터만 선택합니다.
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Residual_Ratio', 'speed', 'acceleration'])
    data = data[data['Residual_Ratio'] != 0]

    # 로그 스케일을 적용하기 위해 Residual_Ratio의 절대값에 로그 변환을 적용합니다.
    data['Residual_Ratio_Log'] = np.log10(np.abs(data['Residual_Ratio']) + 1e-6)

    # Residual_Ratio의 부호를 색상으로 표현하기 위한 컬럼 추가
    data['Residual_Sign'] = data['Residual_Ratio'].apply(lambda x: 'Positive' if x > 0 else 'Negative')

    # 데이터 샘플링
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)

    # Plotly 3D 산점도 생성
    fig = px.scatter_3d(
        data,
        x='speed',
        y='acceleration',
        z='Residual_Ratio_Log',
        color='Residual_Sign',
        title='3D Scatter Plot of Residual Ratio (Plotly)',
        labels={
            'speed': 'Speed (km/h)',
            'acceleration': 'Acceleration (m/s²)',
            'Residual_Ratio_Log': 'log10(|Residual Ratio|)'
        },
        opacity=0.7,
        color_discrete_map={'Positive': 'blue', 'Negative': 'red'}
    )

    fig.update_layout(scene=dict(
        xaxis_title='Speed (km/h)',
        yaxis_title='Acceleration (m/s²)',
        zaxis_title='log10(|Residual Ratio|)'
    ))

    fig.show()

def plot_residual_ratio_pdf_separated(data, exclude_percent=0.1):
    """
    Residual_Ratio의 양수와 음수 분포를 각각 히스토그램과 KDE로 시각화합니다.
    상위 exclude_percent%와 하위 exclude_percent%의 데이터를 제외합니다.
    """
    # NaN 및 무한대 값을 제거하고 Residual_Ratio가 0이 아닌 데이터만 선택
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Residual_Ratio'])
    data = data[data['Residual_Ratio'] != 0]

    # 상위 0.1%와 하위 0.1% 제거
    lower_bound = data['Residual_Ratio'].quantile(exclude_percent / 100)
    upper_bound = data['Residual_Ratio'].quantile(1 - exclude_percent / 100)
    filtered_data = data[(data['Residual_Ratio'] >= lower_bound) & (data['Residual_Ratio'] <= upper_bound)]

    plt.figure(figsize=(12, 6))

    # 양수와 음수로 분리
    positive = filtered_data[filtered_data['Residual_Ratio'] > 0]['Residual_Ratio']
    negative = filtered_data[filtered_data['Residual_Ratio'] < 0]['Residual_Ratio']

    # 히스토그램과 KDE 플롯
    sns.histplot(positive, bins=100, kde=True, stat="density",
                 color='blue', edgecolor='black', alpha=0.6, label='Positive Residual Ratio')
    sns.histplot(negative, bins=100, kde=True, stat="density",
                 color='red', edgecolor='black', alpha=0.6, label='Negative Residual Ratio')

    plt.title(f'Residual Ratio Distribution (positive vs negative) (excluding upper/lower {exclude_percent}%)')
    plt.xlabel('Residual Ratio')
    plt.ylabel('Density')
    plt.legend()
    # 옵션: y축을 로그 스케일로 설정 (필요 시 주석 해제)
    # plt.yscale('log')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def plot_residual_ratio_pdf(data, exclude_percent=0.1):
    """
    Residual_Ratio의 전체 분포를 히스토그램과 KDE로 시각화합니다.
    상위 exclude_percent%와 하위 exclude_percent%의 데이터를 제외합니다.
    """
    # NaN 및 무한대 값을 제거하고 Residual_Ratio가 0이 아닌 데이터만 선택
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Residual_Ratio'])
    data = data[data['Residual_Ratio'] != 0]

    # 상위 0.1%와 하위 0.1% 제거
    lower_bound = data['Residual_Ratio'].quantile(exclude_percent / 100)
    upper_bound = data['Residual_Ratio'].quantile(1 - exclude_percent / 100)
    filtered_data = data[(data['Residual_Ratio'] >= lower_bound) & (data['Residual_Ratio'] <= upper_bound)]

    plt.figure(figsize=(12, 6))

    # 히스토그램과 KDE 플롯
    sns.histplot(filtered_data['Residual_Ratio'], bins=100, kde=True, stat="density",
                 color='skyblue', edgecolor='black', alpha=0.6)

    plt.title(f'Residual Ratio Distribution (excluding upper/lower {exclude_percent}% )')
    plt.xlabel('Residual Ratio')
    plt.ylabel('Density')
    # 옵션: y축을 로그 스케일로 설정 (필요 시 주석 해제)
    # plt.yscale('log')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def plot_power_phys_pdf(data, exclude_percent=0.1):
    # NaN 및 무한대 값을 제거합니다.
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Power_phys', 'Residual'])

    # 상위 1%와 하위 1% 제거
    lower_bound = data['Residual'].quantile(exclude_percent / 100)
    upper_bound = data['Residual'].quantile(1 - exclude_percent / 100)
    filtered_data = data[(data['Residual'] >= lower_bound) & (data['Residual'] <= upper_bound)]

    plt.figure(figsize=(12, 6))

    # 히스토그램과 KDE 플롯
    sns.histplot(filtered_data['Residual'], bins=100, kde=True, stat="density",
                 color='green', edgecolor='black', alpha=0.6)

    plt.title('Residual Distribution (excluding upper/lower 0.1%)')
    plt.xlabel('Residual')
    plt.ylabel('Density')
    # x축 범위를 필요에 따라 조정하세요
    plt.xlim(filtered_data['Residual'].min(), filtered_data['Residual'].max())
    # 옵션: y축을 로그 스케일로 설정 (필요 시 주석 해제)
    # plt.yscale('log')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def main():
    # CSV 파일들이 있는 디렉토리 경로를 설정합니다.
    directory = r'D:\SamsungSTF\Processed_Data\TripByTrip'

    selected_car = 'EV6'

    # 선택한 차종에 해당하는 단말기 번호(디바이스 ID)를 가져옵니다.
    device_ids = vehicle_dict[selected_car]

    # 디렉토리에서 모든 CSV 파일을 가져옵니다.
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

    # 단말기 번호가 파일명에 포함된 파일만 선택합니다.
    files = []
    for file in all_files:
        filename = os.path.basename(file)
        for device_id in device_ids:
            if device_id in filename:
                files.append(file)
                break  # 다른 단말기 번호를 확인할 필요 없음

    if not files:
        print(f"No files found for the selected vehicle: {selected_car}")
        return

    print(f"Processing {len(files)} files for vehicle: {selected_car}")

    # 파일들을 처리합니다.
    full_data = process_files(files)

    # 분포를 시각화합니다.
    # plot_residual_ratio_distribution(full_data)

    # 3D 분포를 Plotly로 인터랙티브하게 시각화합니다.
    # plot_residual_ratio_3d_plotly(full_data)

    # Residual_Ratio 분포 시각화
    plot_residual_ratio_pdf(full_data)
    plot_residual_ratio_pdf_separated(full_data)
    # plot_power_phys_pdf(full_data)
if __name__ == '__main__':
    main()
