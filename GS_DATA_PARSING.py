import csv
import os

# Set the file path
win_file_path ='D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\'
mac_file_path = '/Users/woojin/Downloads/한국에너지공과대학교_샘플데이터/'
file_path = win_file_path

# Set the save path
win_save_path ='D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\processed\\'
mac_save_path = '/Users/woojin/Downloads/한국에너지공과대학교_샘플데이터/processed/'
save_path = win_save_path


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

        data = []
        last_line = None
        for i, row in enumerate(reader):
            if i == 0 or i == 2:  # skip first and third row
                continue
            if last_line is not None:
                data.append(last_line)
            last_line = row

        # 첫 번째 행에서 각 열의 공백을 제거합니다.
        data[0] = [col.strip() for col in data[0]]

        # ','를 구분자로 사용해 출력 파일을 작성합니다.
        with open(os.path.join(save_path, file[:-4] + "_parsed.csv"), 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerows(data)