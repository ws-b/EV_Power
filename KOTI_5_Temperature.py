import os
import glob
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scipy.spatial import cKDTree
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

# Custom exception for rate limit exceeded
class RateLimitExceededError(Exception):
    """Exception raised when the API rate limit is exceeded."""
    pass

# 1. 데이터 로드 및 전처리를 위한 함수 정의

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    return df

def load_stations(stations_file):
    stations = pd.read_csv(stations_file)
    stations['longitude'] = stations['LON'].astype(float)
    stations['latitude'] = stations['LAT'].astype(float)
    return stations

def find_nearest_stations(df, stations):
    station_coords = stations[['longitude', 'latitude']].values
    tree = cKDTree(station_coords)
    data_coords = df[['longitude', 'latitude']].values
    distances, indices = tree.query(data_coords, k=1)
    df['STN_ID'] = stations.iloc[indices]['STN_ID'].values
    return df

def prepare_unique_requests(df):
    df['tm'] = df['time'].dt.strftime('%Y%m%d%H00')  # 'yyyymmddHH00' 형식
    unique_requests = df[['tm', 'STN_ID']].drop_duplicates()
    return unique_requests

def get_observation_data_text(tm, stn, auth_key, help_param=0):
    url = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php'
    params = {
        'tm': tm,
        'stn': stn,
        'help': help_param,
        'authKey': auth_key
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 403:
            # Rate limit exceeded
            raise RateLimitExceededError(f"Rate limit exceeded for tm={tm}, stn={stn}")
        response.raise_for_status()  # Raise an error for bad status codes
        if not response.text.strip():
            print(f"Empty response for tm={tm}, stn={stn}")
            return None
        return response.text  # Return the response text
    except RateLimitExceededError as e:
        print(e)
        raise  # Re-raise the exception to stop processing
    except requests.exceptions.RequestException as e:
        print(f"Request failed for tm={tm}, stn={stn}: {e}")
        return None

def fetch_observation_data(unique_requests, auth_key, help_param=0, max_workers=10):
    observation_data = {}
    def fetch_and_store_text(row):
        tm = row['tm']
        stn = row['STN_ID']
        data = get_observation_data_text(tm, stn, auth_key, help_param)
        if data:
            observation_data[(tm, stn)] = data
        # Add a small delay to reduce server load
        time.sleep(0.1)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_and_store_text, row) for idx, row in unique_requests.iterrows()]
        for future in as_completed(futures):
            try:
                future.result()
            except RateLimitExceededError as e:
                print(f"Rate limit exceeded: {e}")
                executor.shutdown(wait=False)
                raise e  # Re-raise to stop further processing
            except Exception as e:
                print(f"An error occurred: {e}")
    return observation_data

def parse_observation_text(text_content):
    try:
        lines = text_content.splitlines()
        data_line = None
        for line in lines:
            if line.startswith('#'):
                continue  # Skip comment lines
            if line.strip() == '':
                continue  # Skip empty lines
            data_line = line.strip()
            break  # Found the first data line

        if not data_line:
            print("No data line found in the response.")
            return {'TA': np.nan}

        # Split the data line by whitespace
        fields = data_line.split()

        if len(fields) < 12:
            print("Insufficient number of fields in the data line.")
            return {'TA': np.nan}

        # 'TA' is the 12th field (index 11)
        ta = fields[11]

        # Check if 'TA' is numeric, else set as NaN
        try:
            ta_float = float(ta)
        except ValueError:
            ta_float = np.nan

        return {'TA': ta_float}
    except Exception as e:
        print(f"Error parsing observation text: {e}")
        return {'TA': np.nan}

def parse_all_observations(observation_data):
    parsed_data = {}
    for key, text_content in observation_data.items():
        data = parse_observation_text(text_content)
        parsed_data[key] = data
    return parsed_data

def add_external_temp(df, parsed_data):
    def extract_TA_text(row):
        key = (row['tm'], row['STN_ID'])
        data = parsed_data.get(key, {})
        try:
            return data.get('TA', np.nan)  # Return NaN if 'TA' field is missing
        except (ValueError, TypeError):
            return np.nan

    df['ext_temp'] = df.apply(extract_TA_text, axis=1)
    return df

def save_processed_data(df, original_file):
    # Define columns to save
    columns_tosave = ['time', 'x', 'y', 'longitude', 'latitude', 'speed', 'acceleration', 'ext_temp']

    # Overwrite the original file
    df.to_csv(original_file, columns=columns_tosave, index=False)
    print(f"Data saved successfully: {original_file}")

# 2. 파일별 처리 함수 정의

def process_file(file_path, stations, auth_key, help_param=0):
    print(f"Processing file: {file_path}")

    # Load data
    df = load_data(file_path)

    # Find nearest observation stations
    df = find_nearest_stations(df, stations)

    # Prepare unique API requests
    unique_requests = prepare_unique_requests(df)

    # Fetch observation data via API calls
    observation_data = fetch_observation_data(unique_requests, auth_key, help_param)

    # Parse all API responses
    parsed_data = parse_all_observations(observation_data)

    # Add external temperature data
    df = add_external_temp(df, parsed_data)

    # Overwrite the original file
    save_processed_data(df, file_path)

def load_processed_files(processed_files_path):
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            processed_files = f.read().splitlines()
    else:
        processed_files = []
    return processed_files

def save_processed_file(processed_files_path, file_name):
    with open(processed_files_path, 'a') as f:
        f.write(file_name + '\n')


# Helper function for parallel processing
def check_file_needs_processing(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'ext_temp' not in df.columns:
            return (file_path, True)
        elif df['ext_temp'].isna().any():
            return (file_path, True)
        else:
            return (file_path, False)
    except Exception as e:
        print(f"Error reading file ({file_path}): {e}")
        return (file_path, False)


# 3. 메인 함수 정의
def main():
    # Load .env file
    load_dotenv()
    auth_key = os.getenv('KMA_API_KEY')
    if not auth_key:
        print("Error: KMA_API_KEY not found in environment variables.")
        return

    help_param = 0

    # Define folder paths
    input_folder = r"D:\SamsungSTF\Processed_Data\KOTI"
    stations_file = r"D:\SamsungSTF\Data\KMA\Stations.csv"

    # Load station data (only once)
    stations = load_stations(stations_file)

    # Get list of all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in the folder: {input_folder}")
        return

    print(f"Number of CSV files found: {len(csv_files)}")

    # 병렬로 파일 검사 수행
    files_to_process = []
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        future_to_file = {executor.submit(check_file_needs_processing, file): file for file in csv_files}
        # Iterate over completed tasks with tqdm progress bar
        for future in tqdm(as_completed(future_to_file), total=len(csv_files), desc="Checking files for 'ext_temp'"):
            file, needs_processing = future.result()
            if needs_processing:
                files_to_process.append(file)

    if not files_to_process:
        print("No files require processing (no missing 'ext_temp' values).")
        return

    print(f"Number of CSV files to process: {len(files_to_process)}")

    # Process files one by one
    for file in tqdm(files_to_process, desc="Processing files"):
        try:
            process_file(file, stations, auth_key, help_param)
        except RateLimitExceededError as e:
            print(f"Rate limit exceeded: {e}")
            print("Processing stopped due to rate limit.")
            break  # Stop processing further files
        except Exception as e:
            print(f"Error processing file ({file}): {e}")

    print("All applicable files have been processed.")

if __name__ == "__main__":
    main()
