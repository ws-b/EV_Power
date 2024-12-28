import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 예시 CSV 파일 경로
csv_file = r"D:\SamsungSTF\Processed_Data\TripByTrip\bms_01241227999-2023-05-trip-25.csv"

# CSV 파일 읽기
data = pd.read_csv(csv_file)

# 'time' 열을 datetime으로 변환
data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

# 'elapsed_time' 계산 (초 단위)
data['elapsed_time'] = (data['time'] - data['time'].iloc[0]).dt.total_seconds()

# 이동 윈도우 크기 설정 (예: 5)
window_size = 5

# 롤링 통계량 계산
data['mean_accel_10'] = data['acceleration'].rolling(window=window_size).mean().bfill()
data['std_accel_10'] = data['acceleration'].rolling(window=window_size).std().bfill()
data['mean_speed_10'] = data['speed'].rolling(window=window_size).mean().bfill()
data['std_speed_10'] = data['speed'].rolling(window=window_size).std().bfill()

# -------------------------
# 3.1. Speed와 Acceleration의 멀티 y-축 그래프
# -------------------------

# 그래프 크기 설정
fig, ax1 = plt.subplots(figsize=(12, 5))

# 첫 번째 y축: Speed
color = 'tab:blue'
ax1.set_xlabel('Elapsed Time (seconds)')
ax1.set_ylabel('Speed [km/h]', color=color)
ax1.plot(data['elapsed_time'], data['speed']*3.6, color=color, label='Speed', alpha=0.75)
ax1.tick_params(axis='y', labelcolor=color)

# 두 번째 y축: Acceleration
ax2 = ax1.twinx()  # 두 번째 y축 생성
color = 'tab:red'
ax2.set_ylabel('Acceleration [m/s^2]', color=color)
ax2.plot(data['elapsed_time'], data['acceleration'], color=color, label='Acceleration', alpha=0.75)
ax2.tick_params(axis='y', labelcolor=color)

# 제목 및 레이아웃 설정
plt.title('Speed and Acceleration over Elapsed Time')
fig.tight_layout()

# 범례 추가
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.show()

# -------------------------
# 3.2. Speed의 롤링 통계량 멀티 y-축 그래프
# -------------------------

# 그래프 크기 설정
fig, ax1 = plt.subplots(figsize=(12, 5))

# 첫 번째 y축: Mean Speed
color = 'tab:pink'
ax1.set_xlabel('Elapsed Time (seconds)')
ax1.set_ylabel('Mean Speed (10seconds)', color=color)
ax1.plot(data['elapsed_time'], data['mean_speed_10']*3.6, color=color, linestyle='-', label='Mean Speed (10seconds)', alpha=0.75)
ax1.tick_params(axis='y', labelcolor=color)

# 두 번째 y축: Std Speed
ax2 = ax1.twinx()  # 두 번째 y축 생성
color = 'tab:cyan'
ax2.set_ylabel('Std Speed (10seconds)', color=color)
ax2.plot(data['elapsed_time'], data['std_speed_10']*3.6, color=color, linestyle='--', label='Std Speed (10seconds)', alpha=0.75)
ax2.tick_params(axis='y', labelcolor=color)

# 제목 및 레이아웃 설정
plt.title('Rolling Mean and Std of Speed over Elapsed Time')
fig.tight_layout()

# 범례 추가
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.show()

# -------------------------
# 3.3. Acceleration의 롤링 통계량 멀티 y-축 그래프
# -------------------------

# 그래프 크기 설정
fig, ax1 = plt.subplots(figsize=(12, 5))

# 첫 번째 y축: Mean Acceleration
color = 'tab:green'
ax1.set_xlabel('Elapsed Time (seconds)')
ax1.set_ylabel('Mean Acceleration (10seconds)', color=color)
ax1.plot(data['elapsed_time'], data['mean_accel_10'], color=color, linestyle='-', label='Mean Acceleration (10seconds)', alpha=0.75)
ax1.tick_params(axis='y', labelcolor=color)

# 두 번째 y축: Std Acceleration
ax2 = ax1.twinx()  # 두 번째 y축 생성
color = 'tab:orange'
ax2.set_ylabel('Std Acceleration (10seconds)', color=color)
ax2.plot(data['elapsed_time'], data['std_accel_10'], color=color, linestyle='--', label='Std Acceleration (10seconds)', alpha=0.75)
ax2.tick_params(axis='y', labelcolor=color)

# 제목 및 레이아웃 설정
plt.title('Rolling Mean and Std of Acceleration over Elapsed Time')
fig.tight_layout()

# 범례 추가
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.show()
