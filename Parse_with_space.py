import os
import csv
import datetime

# Get the current date and time
now = datetime.datetime.now()

# Format it as a string
date_string = now.strftime("%Y-%m-%d")

win_file_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\경로데이터 샘플 및 데이터 정의서\1. 포인트 경로 데이터.txt'
mac_file_path = ''
file_path = os.path.normpath(win_file_path)

win_save_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터\230706_2'
mac_save_path = ''
save_path = os.path.normpath(win_save_path)

# check if save_path exists
if not os.path.exists(save_path):
    # if not, create the directory
    os.makedirs(save_path)


# Open the file for reading
with open(file_path, "r") as f:
    # Loop over each line in the file, keeping track of the line number
    for line_num, line in enumerate(f):
        # Split the line into individual words, keeping track of the word number
        words = line.split()
        for word_num, word in enumerate(words):
            # Open a new file for writing, using the line number in the filename
            with open(os.path.join(save_path, f"230704_{line_num+1}.txt"), "w") as wf:
                # Write the current word to the file
                wf.write(word)
