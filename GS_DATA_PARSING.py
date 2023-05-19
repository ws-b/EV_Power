import csv
import os

# Set the file path
win_file_path ='D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\1. 포인트 경로 데이터.txt'
mac_file_path = '/Users/woojin/Downloads/한국에너지공과대학교_샘플데이터/'
file_path = mac_file_path

# Set the save path
win_save_path ='D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터\\'
mac_save_path = '/Users/woojin/Downloads/한국에너지공과대학교_샘플데이터/processed/'
save_path = mac_save_path


def get_file_list(folder_path):
    # 폴더 내의 모든 파일 리스트 가져오기
    file_list = os.listdir(folder_path)
    csv_files = [file for file in file_list if file.endswith('.csv')]
    return csv_files

# 디렉토리 내의 CSV 파일 리스트 가져오기
files = get_file_list(file_path)
files.sort()

# 모든 파일에 대해
for file in files:
    # '|'를 구분자로 사용해 입력 파일을 읽습니다.
    with open(file_path + file, 'r') as infile:
        reader = csv.reader(infile, delimiter='|')

        # ','를 구분자로 사용해 출력 파일을 작성합니다.
        with open(save_path + file[:-4] + "_parsed.csv", 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')

            # 입력 파일의 각 행에 대해
            for row in reader:
                # 출력 파일에 행을 작성합니다.
                writer.writerow(row)