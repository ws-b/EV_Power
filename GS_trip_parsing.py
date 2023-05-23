import csv
import datetime
import os

# 파일이 들어있는 폴더 경로
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\speed-acc\\'
mac_folder_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터/'
win_save_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\trip_by_trip\\'
mac_save_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터 Processed/'

folder_path = win_folder_path
save_path = win_save_path

def get_file_list(folder_path):
    # 폴더 내의 모든 파일 리스트 가져오기
    file_list = os.listdir(folder_path)
    txt_files = []
    for file in file_list:
        if file.endswith('.txt'):
            txt_files.append(file)
    return txt_files

# 파일 리스트 가져오기
files = get_file_list(folder_path)
files.sort()

# txt 파일 열기
for i in range(0, len(files)):
    # csv 파일 읽기
    data = np.genfromtxt('your_file.csv', delimiter=',', skip_header=1)

    # 'time'과 'speed' 열 추출
    time = data[:, 0]
    speed = data[:, 1]

    # 각 행 사이의 시간 간격 계산
    delta_time = np.diff(time, prepend=time[0])

    cut_time = 300 # 정차시간을 5분으로 잡아서 trip parsing
    cut = [0]
    trip[0][0] = 0
    for i in range(1, len(utc_times)):
        if (utc_times[i]-utc_times[i-1]).seconds >= cut_time:
            cut.append(i)
        elif i + 1 == len(utc_times):
            cut.append(i)
    j = 0
    for i in range(1, len(utc_times)):
        if (utc_times[i] - utc_times[i-1]).seconds < cut_time:
            trip[i][0] = (utc_times[i] - utc_times[cut[j]]).seconds
        elif (utc_times[i] - utc_times[i-1]).seconds >= cut_time:
            j += 1
            trip[i][0] = (utc_times[i] - utc_times[cut[j]]).seconds

    userID = items[0]
    m = 1

    # Open a CSV file for writing depending on trip number
    if len(cut) == 2:
        globals()[f"{userID}"] = trip[cut[0]:cut[1]] # Parsing Trip with i series
        with open(save_path + f"{userID}.csv", mode='w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)

            # Write each row of the list to the CSV file
            for row in globals()[f"{userID}"]:
                writer.writerow(row)
        del globals()[f"{userID}"]
    else:
        for i in range(0,len(cut)-1):
            globals()[f"{userID}" + '_' + str(m)] = trip[cut[i]:cut[i+1]] # Parsing Trip with i series
            m += 1
        for i in range(1, m):
            with open(save_path + f"{userID}_{i}.csv", mode='w', newline='') as file:
                # Create a CSV writer object
                writer = csv.writer(file)

                # Write each row of the list to the CSV file
                for row in globals()[f"{userID}_{i}"]:
                    writer.writerow(row)
            del globals()[f"{userID}_{i}"]