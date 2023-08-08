import pandas as pd
import matplotlib.pyplot as plt
import os

# 주어진 폴더에서 특정 확장자를 가진 파일 목록을 반환하는 함수
def get_file_list(folder_path, file_extension='.csv'):
    file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(file_extension)]
    file_lists.sort()
    return file_lists

# 1분 간격으로 데이터를 처리하는 함수
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
        aggregated_data.append({
            'time_start': segment.index.min(),
            'time_end': segment.index.max(),
            'avg_speed': segment['speed'].mean(),
            'avg_ext_temp': segment['ext_temp'].mean(),
            'cumulative_Power_IV': segment['Power_IV'].sum()
        })
    return pd.DataFrame(aggregated_data)


# 주어진 경로에서 파일을 불러와 처리한 후 3D 플롯을 생성하는 함수
def plot_3D_from_directory(folder_path):
    file_lists = get_file_list(folder_path)
    all_results = []
    for file in file_lists:
        data_path = os.path.join(folder_path, file)
        data = pd.read_csv(data_path)
        result = process_data_efficiently(data)
        all_results.append(result)
    final_data = pd.concat(all_results, ignore_index=True)

    # 3D 플롯 생성
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = final_data['avg_speed']
    y = final_data['avg_ext_temp']
    z = final_data['cumulative_Power_IV']
    ax.scatter(x, y, z, c=z, marker='o', cmap='viridis')
    ax.set_xlabel('Average Speed')
    ax.set_ylabel('Average External Temperature')
    ax.set_zlabel('Cumulative Power_IV')
    plt.title('3D Scatter Plot of Trips')
    plt.show()

# 함수 호출
folder_path = r"D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\1min_trip"
plot_3D_from_directory(folder_path)
