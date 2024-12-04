import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from GS_vehicle_dict import vehicle_dict  # 외부 모듈에서 vehicle_dict 가져오기
from tqdm import tqdm  # 진행 표시줄을 위한 tqdm

def get_file_lists(directory):
    """
    주어진 디렉토리에서 vehicle_dict에 정의된 차량들의 CSV 파일 목록을 반환합니다.
    """
    vehicle_files = {vehicle: [] for vehicle in vehicle_dict.keys()}

    # 디렉토리 내의 파일들을 순회
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # 파일 이름이 차량 ID와 일치하는지 확인
            for vehicle, ids in vehicle_dict.items():
                if any(vid in filename for vid in ids):
                    vehicle_files[vehicle].append(os.path.join(directory, filename))
                    break  # 일치하는 차량을 찾으면 다음 파일로 이동

    return vehicle_files

# 함수 사용 예시
directory = r"D:\SamsungSTF\Processed_Data\TripByTrip"
vehicle_files = get_file_lists(directory)
selected_cars = ['EV6', 'Ioniq5']

# 각 차량의 평균 SOH 값을 저장할 딕셔너리 초기화
average_soh = {vehicle: [] for vehicle in selected_cars}

for vehicle in selected_cars:
    files = vehicle_files.get(vehicle, [])
    if not files:
        print(f"No CSV files found for vehicle: {vehicle}")
        continue

    print(f"Processing {vehicle} files...")
    for file in tqdm(files, desc=f"Processing {vehicle}", unit="file"):
        try:
            df = pd.read_csv(file)
            if 'soh' not in df.columns:
                print(f"'soh' column not found in {file}. Skipping this file.")
                continue

            # 'soh' 열을 숫자형으로 변환, 변환 불가능한 값은 NaN으로 설정
            df['soh'] = pd.to_numeric(df['soh'], errors='coerce')
            soh_mean = df['soh'].mean()

            if pd.isna(soh_mean):
                print(f"No valid 'soh' data in {file}. Skipping this file.")
                continue

            average_soh[vehicle].append(soh_mean)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

# **평균 SOH 최소값을 계산하고 출력하는 부분 추가**
print("\n=== 각 차량의 평균 SOH 최소값 ===")
for vehicle in selected_cars:
    soh_values = average_soh.get(vehicle, [])
    if soh_values:
        min_soh = min(soh_values)
        print(f"{vehicle}의 평균 SOH 최소값: {min_soh:.2f}")
    else:
        print(f"{vehicle}에 대한 SOH 데이터가 없습니다.")

# 시각화를 위한 데이터 준비 및 개별 시각화
sns.set(style="white")  # 시각 스타일 설정

for vehicle in selected_cars:
    soh_values = average_soh.get(vehicle, [])
    if not soh_values:
        print(f"No SOH data available to plot for {vehicle}.")
        continue

    plot_df = pd.DataFrame({'Average SOH': soh_values})

    # 시각화 가능한 데이터가 있는지 확인
    if plot_df.empty:
        print(f"No data available to plot for {vehicle}.")
        continue

    # 히스토그램 시각화
    plt.figure(figsize=(12, 7))
    sns.histplot(data=plot_df, x='Average SOH',
                bins=50, color='skyblue', edgecolor='black')

    plt.yscale('log')
    # 그래프 커스터마이징
    plt.title(f'Distribution of Average SOH for {vehicle}', fontsize=16)
    plt.xlabel('Average SOH', fontsize=14)
    plt.ylabel('Number of Trips', fontsize=14)
    plt.tight_layout()
    plt.show()
