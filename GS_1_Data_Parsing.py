import csv
import os
from tqdm import tqdm

win_file_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터'
folder_path = os.path.normpath(win_file_path)

win_save_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\processed'
save_path = os.path.normpath(win_save_path)


# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# 모든 파일에 대해
for file in tqdm(file_lists):
    # '|'를 구분자로 사용해 입력 파일을 읽습니다.
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r') as infile:  # Open the file
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
        if data:  # Make sure data is not empty
            data[0] = [col.strip() for col in data[0]]

        # ','를 구분자로 사용해 출력 파일을 작성합니다.
        with open(os.path.join(save_path, file[:-4] + "_parsed.csv"), 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerows(data)