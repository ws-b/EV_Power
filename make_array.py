import csv
import datetime
import os

win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터'
mac_folder_path = ''
win_save_path = 'G:\공유 드라이브\Battery Software Lab\Data\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터 Processed'
mac_save_path = ''

folder_path = os.path.normpath(win_folder_path)
save_path = os.path.normpath(win_save_path)

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]
file_lists.sort()

# Open each TXT file
for i in range(0, len(file_lists)):
    file_number = file_lists[i]
    file = open(folder_path+f"{file_number}", "r")

    # Read only the first line
    line = file.readline()

    # Split the line into items using comma as the separator
    items = line.split(",")

    # Extract GPS coordinates by excluding the first item and forming a 2D array with 3 columns
    gps = []
    for i in range(1, len(items), 3):
        gps.append(items[i:i+3])

    unix_times = [int(item[0]) for item in gps]
    utc_times = []

    for unix_time in unix_times:
        dt_object = datetime.datetime.fromtimestamp(unix_time)
        utc_time = datetime.datetime.strptime(str(dt_object), '%Y-%m-%d %H:%M:%S')
        utc_times.append(utc_time)

    cut_time = 300  # Set the stationary time to 5 minutes for trip parsing
    cut = [0]
    gps[0][0] = 0
    for i in range(1, len(utc_times)):
        if (utc_times[i]-utc_times[i-1]).seconds >= cut_time:
            cut.append(i)
        elif i + 1 == len(utc_times):
            cut.append(i)
    j = 0
    for i in range(1, len(utc_times)):
        if (utc_times[i] - utc_times[i-1]).seconds < cut_time:
            gps[i][0] = (utc_times[i] - utc_times[cut[j]]).seconds
        elif (utc_times[i] - utc_times[i-1]).seconds >= cut_time:
            j += 1
            gps[i][0] = (utc_times[i] - utc_times[cut[j]]).seconds

    userID = items[0]
    m = 1

    # Open a CSV file for writing depending on the trip number
    if len(cut) == 2:
        globals()[f"{userID}"] = gps[cut[0]:cut[1]]  # Parsing Trip with i series
        with open(save_path + f"{userID}.csv", mode='w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)

            # Write each row of the list to the CSV file
            for row in globals()[f"{userID}"]:
                writer.writerow(row)
        del globals()[f"{userID}"]
    else:
        for i in range(0, len(cut)-1):
            globals()[f"{userID}" + '_' + str(m)] = gps[cut[i]:cut[i+1]]  # Parsing Trip with i series
            m += 1
        for i in range(1, m):
            with open(save_path + f"{userID}_{i}.csv", mode='w', newline='') as file:
                # Create a CSV writer object
                writer = csv.writer(file)

                # Write each row of the list to the CSV file
                for row in globals()[f"{userID}_{i}"]:
                    writer.writerow(row)
            del globals()[f"{userID}_{i}"]
