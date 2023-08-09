import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_file_list(folder_path, file_extension='.csv'):
    file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(file_extension)]
    file_lists.sort()
    return file_lists
"""
def process_data_efficiently(data):
    data['time'] = pd.to_datetime(data['time'])

    # 중복된 시간 인덱스 제거
    data.drop_duplicates(subset='time', inplace=True)

    data.set_index('time', inplace=True)
    data['time_diff'] = data.index.to_series().diff().dt.total_seconds().fillna(0)
    boundaries = data[data['time_diff'] > 60].index
    segments = [data.loc[i:j] for i, j in zip([data.index[0]] + list(boundaries), list(boundaries) + [data.index[-1]])]

    aggregated_data = []
    for segment in segments:
        # Calculate distance using each speed data multiplied by time_diff
        v = segment['speed']
        v = np.array(v)
        t_diff = segment['time_diff']
        t_diff = np.array(t_diff.fillna(0))

        distance = v * t_diff
        total_distance = distance.cumsum()

        bms_power = segment['Power_IV']
        bms_power = np.array(bms_power)
        energy = bms_power * t_diff / 3600 / 1000
        energy_cumulative = energy.cumsum()

        # calculate Total distance / net_discharge for each file (if net_discharge is 0, set the value to 0)
        distance_per_total_energy = (total_distance[-1] / 1000) / energy_cumulative[-1] if energy_cumulative[-1] != 0 else 0

        aggregated_data.append({
            'time_start': segment.index.min(),
            'time_end': segment.index.max(),
            'avg_speed_km_h': segment['speed'].mean() * 3.6,  # Convert m/s to km/h for average speed
            'avg_ext_temp': segment['ext_temp'].mean(),
            'km_per_energy': distance_per_total_energy
        })

    return pd.DataFrame(aggregated_data)
"""


def process_data_efficiently(data):
    data['time'] = pd.to_datetime(data['time'])
    data.set_index('time', inplace=True)

    # Resample data into 5-minute intervals
    data_resampled = data.resample('5T').mean().dropna()

    aggregated_data = []

    for start, segment in data_resampled.groupby(pd.Grouper(freq='5T')):
        # Calculate distance using each speed data multiplied by time_diff
        v = segment['speed']
        if v.empty:  # Check if segment is empty
            continue

        v = np.array(v)

        distance = v * 60  # 60 seconds for each segment
        total_distance = distance.cumsum()

        bms_power = segment['Power_IV']
        if bms_power.empty:  # Check if segment is empty
            continue

        bms_power = np.array(bms_power)
        energy = bms_power * 60 / 3600 / 1000
        energy_cumulative = energy.cumsum()

        # Calculate Total distance / net_discharge for each segment (if net_discharge is 0, set the value to 0)
        distance_per_total_energy = (total_distance[-1] / 1000) / energy_cumulative[-1] if energy_cumulative[
                                                                                               -1] != 0 else 0

        aggregated_data.append({
            'time_start': start,
            'time_end': start + pd.Timedelta(minutes=1),
            'avg_speed_km_h': segment['speed'].mean() * 3.6,  # Convert m/s to km/h for average speed
            'avg_ext_temp': segment['ext_temp'].mean(),
            'km_per_energy': distance_per_total_energy
        })

    return pd.DataFrame(aggregated_data)


def plot_3D_speed(folder_path):
    file_lists = get_file_list(folder_path)
    all_results = []
    total_segments = 0
    for file in file_lists:
        data_path = os.path.join(folder_path, file)
        data = pd.read_csv(data_path)
        result = process_data_efficiently(data)
        all_results.append(result)
        total_segments += len(result)
    final_data = pd.concat(all_results, ignore_index=True)

    # 3D 플롯 생성
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = final_data['avg_speed_km_h']
    y = final_data['avg_ext_temp']
    z = final_data['km_per_energy']

    # 속도에 따른 색상 지정
    colors = final_data['avg_speed_km_h'].apply(assign_color_by_speed)

    ax.scatter(x, y, z, c=colors, marker='o')
    ax.set_xlabel('Average Speed (km/h)')
    ax.set_ylabel('Average External Temperature')
    ax.set_zlabel('km per Energy (kWh)')
    plt.xlim(0, 110)
    plt.ylim(-10, 40)
    ax.set_zlim(0, 10)
    plt.title('3D Scatter Plot of Trips')
    # Add segment count as an annotation inside the graph
    ax.text(50, 30, 8, f"Segments: {total_segments}", color='black', backgroundcolor='white')


    plt.tight_layout()
    plt.show()



def assign_color_by_speed(speed):
    if 0 <= speed <= 10:
        return 'red'
    elif 11 <= speed <= 20:
        return 'orange'
    elif 21 <= speed <= 30:
        return 'yellow'
    elif 31 <= speed <= 40:
        return 'green'
    elif 41 <= speed <= 50:
        return 'blue'
    elif 51 <= speed <= 60:
        return 'purple'
    elif 61 <= speed <= 70:
        return 'cyan'
    elif 71 <= speed <= 80:
        return 'magenta'
    elif 81 <= speed <= 90:
        return 'brown'
    elif 91 <= speed <= 100:
        return 'pink'
    elif 101 <= speed <= 110:
        return 'gray'
    else:
        return 'black'

def plot_2D_speed(folder_path):
    file_lists = get_file_list(folder_path)
    all_results = []
    for file in file_lists:
        data_path = os.path.join(folder_path, file)
        data = pd.read_csv(data_path)
        result = process_data_efficiently(data)
        all_results.append(result)
    final_data = pd.concat(all_results, ignore_index=True)

    # 2D 플롯 생성
    colors = final_data['avg_speed_km_h'].apply(assign_color_by_speed)

    plt.figure(figsize=(10, 8))
    plt.scatter(final_data['avg_ext_temp'], final_data['km_per_energy'], c=colors)
    plt.xlabel('Average External Temperature')
    plt.ylabel('km per Energy (kWh)')
    plt.title('2D Scatter Plot of Trips')
    plt.ylim(0, 10)
    plt.tight_layout()
    plt.show()

# 함수 호출
folder_path = r"D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\1min_trip"
plot_3D_speed(folder_path)
plot_2D_speed(folder_path)