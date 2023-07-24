import os
import csv
from tqdm import tqdm

def get_file_list(folder_path, file_extension='.csv'):
    file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(file_extension)]
    file_lists.sort()
    return file_lists
def parse_spacebar(file_lists, folder_path, save_path):
    for file in tqdm(file_lists):
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
    print("Done!")