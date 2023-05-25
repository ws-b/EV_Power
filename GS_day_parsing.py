import pandas as pd

# 파일이 들어있는 폴더 경로
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\speed-acc\\'
mac_folder_path = '/Users/woojin/Documents/켄텍 자료/삼성미래과제/한국에너지공과대학교_샘플데이터/speed-acc/'
win_save_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\trip_by_trip\\'
mac_save_path = '/Users/woojin/Documents/켄텍 자료/삼성미래과제/한국에너지공과대학교_샘플데이터/day_by_day/'

folder_path = mac_folder_path
save_path = mac_save_path

file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
for file_list in file_lists:
    file = open(folder_path + file_list, "r")

    # csv 파일을 DataFrame으로 읽어오기
    data = pd.read_csv(file)

    # 'DATETIME' 컬럼을 datetime 타입으로 변환
    data['time'] = pd.to_datetime(data['time'])

    # 'DATETIME'에서 일자만 추출하여 'DATE'라는 새로운 컬럼 생성
    data['DATE'] = data['time'].dt.date

    # 일자별로 데이터 분할 후 저장
    for date, group in data.groupby('DATE'):
        # 'DATE' 컬럼 삭제
        group = group.drop(columns=['DATE'])

        # 파일 저장
        group.to_csv(f"{save_path}/{file_list[:-6]}{date}.csv", index=False)