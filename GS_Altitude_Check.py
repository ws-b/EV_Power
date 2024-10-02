import os
import glob
import pandas as pd
import requests
from GS_vehicle_dict import vehicle_dict
from dotenv import load_dotenv
from tqdm import tqdm
# Google Maps API key (replace with your own key)
load_dotenv()
API_KEY = os.getenv('GOOGLE_API_KEY')

# CSV 파일들이 있는 디렉토리 경로를 설정합니다.
directory = r'D:\SamsungSTF\Processed_Data\TripByTrip'
selected_car = 'EV6'

device_ids = vehicle_dict.get(selected_car, [])

if not device_ids:
    print(f"No device IDs found for the selected vehicle: {selected_car}")
    exit()

# 디렉토리에서 'bms'와 'altitude'가 포함된 모든 CSV 파일을 가져옵니다.
all_files = glob.glob(os.path.join(directory, '*bms*altitude*09*.csv'))

# 단말기 번호가 파일명에 포함된 파일만 선택합니다.
files = [file for file in all_files if any(device_id in os.path.basename(file) for device_id in device_ids)]

if not files:
    print(f"No files found for the selected vehicle: {selected_car}")
    exit()

print(f"Processing {len(files)} files for vehicle: {selected_car}")

# Function to fetch altitudes in batches with proper mapping to original indices
def fetch_altitudes_google(lat_lng_batch):
    # Format the lat, lng pairs as 'lat,lng' strings
    locations = '|'.join([f"{lat:.6f},{lng:.6f}" for lat, lng in lat_lng_batch])

    url = f'https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={API_KEY}'

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error with the request: {response.text}")
        return [None] * len(lat_lng_batch)

    result = response.json()
    if 'results' in result:
        return [res['elevation'] for res in result['results']]
    else:
        return [None] * len(lat_lng_batch)

# Define the output directory for error files
output_directory = r'C:\Users\BSL\Desktop\Error'
os.makedirs(output_directory, exist_ok=True)  # Ensure the directory exists

non_100_similarity_files = []

for file in tqdm(files[:100]):
    df = pd.read_csv(file)

    if 'lat' not in df.columns or 'lng' not in df.columns:
        print(f"Skipping file {file} because 'lat' or 'lng' columns are missing.")
        continue

    # Identify rows where both 'lat' and 'lng' exist
    valid_lat_lng_rows = df.dropna(subset=['lat', 'lng'])
    lat_lng_pairs = valid_lat_lng_rows[['lat', 'lng']].values
    indices = valid_lat_lng_rows.index  # Save the original indices

    batch_size = 500  # Adjust batch size if needed
    altitudes_google = []

    for i in range(0, len(lat_lng_pairs), batch_size):
        batch = lat_lng_pairs[i:i + batch_size]
        altitudes_google.extend(fetch_altitudes_google(batch))

    # Map the altitudes back to the original dataframe using the saved indices
    df.loc[indices, 'altitude_google'] = altitudes_google

    # Check if 'altitude' column exists
    if 'altitude' not in df.columns:
        print(f"Skipping comparison for file {file} because 'altitude' column is missing.")
        continue

    # Compare 'altitude' with 'altitude_google' where both are available
    comparison_df = df.dropna(subset=['altitude', 'altitude_google'])

    if len(comparison_df) > 0:
        similarity_ratio = (comparison_df['altitude'].round(2) == comparison_df['altitude_google'].round(2)).mean()
        if similarity_ratio < 1.0:  # Store files with similarity less than 100%
            non_100_similarity_files.append((file, similarity_ratio))
            # Save the modified file with 'altitude_google' column to the output directory
            output_file_path = os.path.join(output_directory, os.path.basename(file))  # Preserve original file name
            df.to_csv(output_file_path, index=False)
    else:
        print(f"No valid rows to compare in file {file}.")

# Output the list of files with similarity less than 100%
if non_100_similarity_files:
    print("Files saved with Similarity Ratio less than 100%:")
    for file, ratio in non_100_similarity_files:
        print(f"File: {file}, Similarity Ratio: {ratio:.2%}")
else:
    print("All files have 100% similarity.")