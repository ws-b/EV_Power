import pandas as pd

# Folder path containing the files
win_folder_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\speed-acc\\'
mac_folder_path = '/Users/woojin/Documents/켄텍 자료/삼성미래과제/한국에너지공과대학교_샘플데이터/speed-acc/'
win_save_path = 'D:\\Data\\대학교 자료\\켄텍 자료\\삼성미래과제\\한국에너지공과대학교_샘플데이터\\trip_by_trip\\'
mac_save_path = '/Users/woojin/Documents/켄텍 자료/삼성미래과제/한국에너지공과대학교_샘플데이터/day_by_day/'

folder_path = mac_folder_path
save_path = mac_save_path

file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
for file_list in file_lists:
    file = open(folder_path + file_list, "r")

    # Read the CSV file into a DataFrame
    data = pd.read_csv(file)

    # Convert 'DATETIME' column to datetime type
    data['time'] = pd.to_datetime(data['time'])

    # Extract the date from 'DATETIME' and create a new column 'DATE'
    data['DATE'] = data['time'].dt.date

    # Split and save the data by date
    for date, group in data.groupby('DATE'):
        # Drop the 'DATE' column
        group = group.drop(columns=['DATE'])

        # Save the file
        group.to_csv(f"{save_path}/{file_list[:-6]}{date}.csv", index=False)
