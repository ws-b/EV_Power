import os
import pandas as pd
from tqdm import tqdm

# Define paths
GPS_ALTITUDE_DIR = r'D:\SamsungSTF\Data\GSmbiz\gps_altitude'
BMS_MERGED_DIR = r'D:\SamsungSTF\Processed_Data\Merged'

def parse_time_column(df, file_path, file_type):
    """
    Parse the 'time' column in the DataFrame using the specified date format.
    If parsing fails, return None.

    :param df: pandas DataFrame
    :param file_path: Path to the file (for logging)
    :param file_type: 'bms' or 'gps' to determine the date format
    :return: DataFrame with parsed 'time' or None if parsing fails
    """
    try:
        if file_type == 'bms':
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='raise')
        elif file_type == 'gps':
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='raise')
        else:
            print(f"Unknown file type '{file_type}' for file {file_path}. Skipping this file.")
            return None
        return df
    except ValueError:
        print(f"Time format error in file {file_path}. Expected format: {'BMS' if file_type == 'bms' else 'GPS'}")
        return None  # Indicate failure


def parse_device_and_date_from_bms_filename(filename):
    """
    Extract device ID and year-month from BMS filename.
    Example filenames:
    - bms_altitude_01241597802_2024-04.csv
    """
    base = os.path.splitext(filename)[0]  # Remove .csv extension
    parts = base.split('_')

    if len(parts) < 4:
        return None, None  # Handle incorrect filenames

    # Assuming the format is 'bms_altitude_{device_id}_{year-month}'
    device_id = parts[2]

    # The last part should contain the date (format: YYYY-MM)
    date_part = parts[3]
    if '-' not in date_part:
        return device_id, None  # Return device ID, but no valid date

    year_month = date_part
    return device_id, year_month


def find_gps_files(device_id, year_month):
    """
    Find GPS altitude files matching the device ID and year-month.
    GPS files are located at GPS_ALTITUDE_DIR/device_id/year-month/*.csv
    """
    gps_folder = os.path.join(GPS_ALTITUDE_DIR, device_id, year_month)
    if not os.path.exists(gps_folder):
        return []
    return [os.path.join(gps_folder, f) for f in os.listdir(gps_folder) if f.endswith('.csv')]


def load_bms_data(bms_file):
    """
    Load BMS data and parse 'time' column.
    """
    try:
        bms_df = pd.read_csv(bms_file)
        bms_df = parse_time_column(bms_df, bms_file, 'bms')
        if bms_df is None:
            return None  # Indicate failure due to time parsing
    except Exception as e:
        print(f"Error reading BMS file {bms_file}: {e}")
        return None  # Skip files that cause errors

    # Initialize 'lat' and 'lng' columns if they don't exist
    if 'lat' not in bms_df.columns:
        bms_df['lat'] = pd.NA
    if 'lng' not in bms_df.columns:
        bms_df['lng'] = pd.NA

    return bms_df


def process_gps_file(bms_df, gps_file, tolerance_seconds=1.5):
    """
    Merge a single GPS file into the BMS DataFrame based on nearest time within tolerance.

    :param bms_df: pandas DataFrame of BMS data
    :param gps_file: Path to the GPS file
    :param tolerance_seconds: Time tolerance in seconds for matching
    :return: Updated BMS DataFrame
    """
    try:
        gps_df = pd.read_csv(gps_file)
        gps_df = parse_time_column(gps_df, gps_file, 'gps')
        if gps_df is None:
            print(f"Skipping GPS file due to time parsing error: {gps_file}")
            return bms_df  # Return BMS DataFrame unchanged

    except Exception as e:
        print(f"Error reading GPS file {gps_file}: {e}")
        return bms_df  # Return BMS DataFrame unchanged

    # Sort GPS data by time
    gps_df = gps_df.sort_values('time').reset_index(drop=True)

    # Perform an asof merge to find nearest GPS point for each BMS time
    merged_df = pd.merge_asof(
        bms_df.sort_values('time'),
        gps_df[['time', 'lat', 'lng']],
        on='time',
        direction='backward',  # 이전 GPS 데이터만 매칭
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
        suffixes=('', '_gps')
    )

    # Count how many GPS points were matched
    matched = merged_df['lat_gps'].notna().sum()
    print(f"GPS file '{os.path.basename(gps_file)}' matched {matched} BMS records within {tolerance_seconds} seconds.")

    # Update 'lat' and 'lng' columns where GPS data is available and not already filled
    # Only fill if the current 'lat'/'lng' is NaN
    mask_lat = merged_df['lat_gps'].notna() & bms_df['lat'].isna()
    mask_lng = merged_df['lng_gps'].notna() & bms_df['lng'].isna()

    bms_df.loc[mask_lat, 'lat'] = merged_df.loc[mask_lat, 'lat_gps']
    bms_df.loc[mask_lng, 'lng'] = merged_df.loc[mask_lng, 'lng_gps']

    filled_lat = mask_lat.sum()
    filled_lng = mask_lng.sum()
    print(f"Filled {filled_lat} 'lat' and {filled_lng} 'lng' from GPS file '{os.path.basename(gps_file)}'.")

    return bms_df


def main():
    desired_columns = [
        'time', 'speed', 'acceleration', 'ext_temp', 'int_temp', 'soc', 'soh',
        'chrg_cable_conn', 'altitude', 'lat', 'lng', 'pack_volt', 'pack_current',
        'Power_data'
    ]
    # Iterate through car type folders in BMS_MERGED_DIR
    for car_type in os.listdir(BMS_MERGED_DIR):
        car_type_path = os.path.join(BMS_MERGED_DIR, car_type)
        if not os.path.isdir(car_type_path):
            continue

        # Iterate through BMS files in the car type folder
        # Only process files that start with 'bms_altitude_' and end with '.csv'
        bms_files = [f for f in os.listdir(car_type_path) if f.endswith('.csv') and f.startswith('bms_altitude_')]
        if not bms_files:
            print(f"No 'bms_altitude_' files found in {car_type_path}. Skipping this folder.")
            continue

        for bms_file in tqdm(bms_files, desc=f'Processing {car_type}'):
            bms_file_path = os.path.join(car_type_path, bms_file)

            # Parse device ID and year-month from filename
            device_id, year_month = parse_device_and_date_from_bms_filename(bms_file)
            if not device_id or not year_month:
                print(f"Skipping file with unexpected name format: {bms_file}")
                continue

            # Find corresponding GPS files
            gps_files = find_gps_files(device_id, year_month)
            if not gps_files:
                print(f"No GPS files found for device {device_id} and period {year_month}.")
                continue

            # Load BMS data
            bms_df = load_bms_data(bms_file_path)
            if bms_df is None:
                print(f"Skipping BMS file due to time parsing error: {bms_file}")
                continue

            initial_na_lat = bms_df['lat'].isna().sum()
            initial_na_lng = bms_df['lng'].isna().sum()

            # Iterate through GPS files and merge them one by one
            for gps_file in gps_files:
                bms_df = process_gps_file(bms_df, gps_file)

            final_na_lat = bms_df['lat'].isna().sum()
            final_na_lng = bms_df['lng'].isna().sum()
            filled_lat = initial_na_lat - final_na_lat
            filled_lng = initial_na_lng - final_na_lng
            print(f"Filled a total of {filled_lat} 'lat' and {filled_lng} 'lng' for BMS file '{bms_file}'.")

            # After processing all GPS files, sort BMS data by time
            bms_df = bms_df.sort_values('time').reset_index(drop=True)

            # Reorder the columns
            bms_df = bms_df[desired_columns]

            # Define output path
            output_car_type_path = os.path.join(BMS_MERGED_DIR, car_type)
            os.makedirs(output_car_type_path, exist_ok=True)
            output_file_path = os.path.join(output_car_type_path, bms_file)

            # Save merged data
            try:
                bms_df.to_csv(output_file_path, index=False)
                print(f"Successfully saved merged file to {output_file_path}\n")
            except Exception as e:
                print(f"Error saving merged file {output_file_path}: {e}")


if __name__ == "__main__":
    main()
