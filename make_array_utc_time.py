import csv
import datetime
import os

# Folder paths
win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터'
win_save_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터 Processed'

folder_path = os.path.normpath(win_folder_path)
save_path = os.path.normpath(win_save_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]
file_lists.sort()

# Open and process each txt file
for i in range(0, len(file_lists)):
    file_number = file_lists[i]
    file = open(folder_path + f"{file_number}", "r")

    # Read only the first line
    line = file.readline()

    # Split the line into a list of items separated by commas
    items = line.split(",")

    # Extract GPS coordinates by excluding the first item and forming a 2D array with 3 columns
    gps = []
    for i in range(1, len(items), 3):
        gps.append(items[i:i + 3])

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
        if (utc_times[i] - utc_times[i - 1]).seconds >= 300:
            cut.append(i)
        elif i + 1 == len(utc_times):
            cut.append(i)

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
        for i in range(0, len(cut) - 1):
            globals()[f"{userID}" + '_' + str(m)] = gps[cut[i]:cut[i + 1]]  # Store trip data in a variable
            m += 1
        for i in range(1, m):
            with open(save_path + f"{userID}_{i}.csv", mode='w', newline='') as file:
                writer = csv.writer(file)  # Create a CSV writer object
                for row in globals()[f"{userID}_{i}"]:
                    writer.writerow(row)  # Write each row of the list to the CSV file
            del globals()[f"{userID}_{i}"]
