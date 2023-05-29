import csv
import datetime

# Folder paths
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터\\'
mac_folder_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터/'
win_save_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\경로데이터 샘플 및 데이터 정의서\\포인트 경로 데이터 Processed\\'
mac_save_path = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터 Processed/'

folder_path = win_folder_path
save_path = win_save_path

# Open the data.txt file
i = 4559
file_number = 'pointdata_'+f"{i}"
file = open(folder_path + f"{file_number}.txt", "r")

# Read only the first line
line = file.readline()

# Split the line into items using comma as the separator
items = line.split(",")

# Extract GPS coordinates by excluding the first item and forming a 2D array with 3 columns
gps = []
for i in range(1, len(items), 3):
    gps.append(items[i:i+3])

# Convert Unix timestamps to UTC datetime objects
unix_times = [int(item[0]) for item in gps]
utc_times = []

for unix_time in unix_times:
    dt_object = datetime.datetime.fromtimestamp(unix_time)
    utc_time = datetime.datetime.strptime(str(dt_object), '%Y-%m-%d %H:%M:%S')
    utc_times.append(utc_time)

# Determine the indices to split the data into trips
cut = [0]
gps[0][0] = 0
for i in range(1, len(utc_times)):
    if (utc_times[i] - utc_times[i-1]).seconds >= 300:
        cut.append(i)
    elif i + 1 == len(utc_times):
        cut.append(i)

# Assign seconds elapsed from the start of each trip to the corresponding GPS data
j = 0
for i in range(1, len(utc_times)):
    if (utc_times[i] - utc_times[i-1]).seconds < 300:
        gps[i][0] = (utc_times[i] - utc_times[cut[j]]).seconds
    elif (utc_times[i] - utc_times[i-1]).seconds >= 300:
        j += 1
        gps[i][0] = (utc_times[i] - utc_times[cut[j]]).seconds

# Extract the user ID and trip number
userID = items[0]
m = 1

# Open a CSV file for writing depending on the number of trips
if len(cut) == 2:
    globals()[f"{userID}"] = gps[cut[0]:cut[1]]  # Store trip data in a variable
    with open(save_path + f"{userID}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)  # Create a CSV writer object
        for row in globals()[f"{userID}"]:
            writer.writerow(row)  # Write each row of the list to the CSV file
    del globals()[f"{userID}"]
else:
    for i in range(0, len(cut)-1):
        globals()[f"{userID}" + '_' + str(m)] = gps[cut[i]:cut[i+1]]  # Store trip data in a variable
        m += 1
    for i in range(1, m):
        with open(save_path + f"{userID}_{i}.csv", mode='w', newline='') as file:
            writer = csv.writer(file)  # Create a CSV writer object
            for row in globals()[f"{userID}_{i}"]:
                writer.writerow(row)  # Write each row of the list to the CSV file
        del globals()[f"{userID}_{i}"]

