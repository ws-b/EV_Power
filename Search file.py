import csv
import datetime

# data.txt 파일 열기
userID = []
for i in range(0, 7418):
    file_number = 'pointdata_'+f"{i}"
    file = open('/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터/'+f"{file_number}.txt", "r")

    # 첫 번째 줄만 읽기
    line = file.readline()

    # 쉼표(,)로 구분된 항목들을 리스트로 변환하기
    items = line.split(",")
    userID.append(items[0])

find = '481'

index_of_item = userID.index(find)
print(index_of_item)