import pandas as pd

# Folder path containing the files
win_folder_path = 'G:\공유 드라이브\Battery Software Lab\Data\한국에너지공과대학교_샘플데이터\speed-acc'

win_save_path = 'G:\공유 드라이브\Battery Software Lab\Data\한국에너지공과대학교_샘플데이터\trip_by_trip'

folder_path = os.path.normpath(win_folder_path)
save_path = os.path.normpath(win_save_path)

file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
for file in file_lists:
    file_path = os.path.join(folder_path, file)

    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)

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
