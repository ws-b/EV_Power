import os
import csv
import datetime

# Get the current date and time
now = datetime.datetime.now()
# Format it as a string
date_string = now.strftime("%y%m%d")

win_file_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터\230706'
folder_path = os.path.normpath(win_file_path)

win_save_path = r'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\경로데이터 샘플 및 데이터 정의서\포인트 경로 데이터_후처리'
save_path = os.path.join(os.path.normpath(win_save_path), date_string)

# check if save_path exists
if not os.path.exists(save_path):
    # if not, create the directory
    os.makedirs(save_path)

# Define start and end years
start_year = 2010
end_year = 2023

# Convert years to Unix timestamps
start_time = datetime.datetime(start_year, 1, 1).timestamp()
end_time = datetime.datetime(end_year, 12, 31, 23, 59, 59).timestamp()

# get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]
file_lists.sort()

# Open each TXT file
for file in file_lists:
    data = open(os.path.join(folder_path, file), "r")

    # Read only the first line
    line = data.readline()

    # Split the line into items using comma as the separator
    items = line.split(",")

    # Extract GPS coordinates by excluding the first item and forming a 2D array with 3 columns
    gps = []
    for i in range(8, len(items), 13):
        gps.append(items[i:i+13])

    # Convert Unix times to datetime objects, and filter out rows not within 2010~2023
    gps = [row for row in gps if start_time <= int(row[0]) <= end_time]

    unix_times = [int(item[0]) for item in gps]
    utc_times = []

    for unix_time in unix_times:
        dt_object = datetime.datetime.fromtimestamp(unix_time)
        utc_time = datetime.datetime.strptime(str(dt_object), "%Y-%m-%d %H:%M:%S")
        utc_times.append(utc_time)

    cut_time = 300  # Set the stop time to 5 minutes for trip parsing
    cut = [0]

    for i in range(1, len(utc_times)):
        if (utc_times[i]-utc_times[i-1]).seconds >= cut_time:
            cut.append(i)
        elif i + 1 == len(utc_times):
            cut.append(i)

    for i in range(0, len(utc_times)):
        gps[i][0] = utc_times[i]

    userID = items[0]
    m = 1

    # Open a CSV file for writing depending on the trip number
    if len(cut) == 2:
        globals()[f"{userID}"] = gps[cut[0]:cut[1]]  # Parsing Trip with i series
        with open(os.path.join(save_path, f"{userID}.csv"), mode='w', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)

            # Write each row of the list to the CSV file
            for row in globals()[f"{userID}"]:
                writer.writerow(row)
        print(f"{userID} is done!")
        del globals()[f"{userID}"]

    else:
        for i in range(0, len(cut)-1):
            globals()[f"{userID}" + '_' + str(m)] = gps[cut[i]:cut[i+1]]  # Parsing Trip with i series
            m += 1
        for i in range(1, m):
            with open(os.path.join(save_path, f"{userID}_{i}.csv"), mode='w', newline='') as file:
                # Create a CSV writer object
                writer = csv.writer(file)

                # Write each row of the list to the CSV file
                for row in globals()[f"{userID}_{i}"]:
                    writer.writerow(row)
            print(f"{userID}_{i} is done!")
            del globals()[f"{userID}_{i}"]
