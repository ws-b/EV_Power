import csv
import datetime

# Open data.txt file
userID = []
for i in range(0, 7418):
    file_number = 'pointdata_'+f"{i}"
    file = open('/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터/'+f"{file_number}.txt", "r")

    # Read only the first line
    line = file.readline()

    # Convert comma-separated items into a list
    items = line.split(",")
    userID.append(items[0])

find = '481'

# Find the index of the item
index_of_item = userID.index(find)
print(index_of_item)
