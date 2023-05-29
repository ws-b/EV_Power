import os
import matplotlib.dates as mdates
import pandas as pd

# 파일이 들어있는 폴더 경로
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\kona_ev\\'
mac_folder_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터 속도-가속도 처리'

folder_path = win_folder_path

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# 파일들을 돌면서 그래프 그리기
for file_list in file_lists:
    # 파일 경로 생성하기
    file_path = os.path.join(folder_path, file_list)
    data = pd.read_csv(file_path)

    # 시간, Power, CHARGE, DISCHARGE 추출
    t = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    CHARGE = data['trip_chrg_pw'].tolist()
    DISCHARGE = data['trip_dischrg_pw'].tolist()

    # CHARGE와 DISCHARGE의 차이 계산
    net_charge = np.array(DISCHARGE) - np.array(CHARGE)

    # Power 데이터를 kWh로 변환 후 누적 계산하기
    Power_kWh = data['Power'] * 0.00055556  # convert kW to kWh considering the 2-second time interval
    Power_kWh_cumulative = Power_kWh.cumsum()

    # 시간 범위가 5분 이상인 경우만 그래프 그리기
    time_range = t.iloc[-1] - t.iloc[0]
    if time_range.total_seconds() >= 300:  # 5 minutes = 300 seconds

        # 그래프 그리기
        fig, ax1 = plt.subplots(figsize=(8, 6))  # set the size of the graph

        color = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Power (kWh)', color=color)
        ax1.plot(t, Power_kWh_cumulative, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()

        color = 'tab:red'
        # we already handled the x-label with ax1
        ax2.set_ylabel('Net Charge (Discharge - Charge) (kWh)', color=color)
        ax2.plot(t, net_charge, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # format the ticks
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # change to display only time
        ax1.xaxis.set_major_locator(mdates.HourLocator())

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        # add date and file name
        date = t.iloc[0].strftime('%Y-%m-%d')
        plt.text(1, 1, date, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', color='black')
        plt.text(0, 1, 'File: ' + file_list, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black')

        # set y limit based on the range of both datasets
        min_val = min(min(Power_kWh_cumulative), min(net_charge))
        max_val = max(max(Power_kWh_cumulative), max(net_charge))
        ax1.set_ylim(min_val, max_val)
        ax2.set_ylim(min_val, max_val)

        plt.title('Cumulative Power (kWh) and Net Charge over Time')
        plt.show()