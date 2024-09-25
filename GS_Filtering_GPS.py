import os
import glob
import pandas as pd
from tqdm import tqdm
import logging

# ================================
# Configuration
# ================================

# Directory containing the CSV files
directory = r'D:\SamsungSTF\Processed_Data\TripByTrip'

# Pattern to match files containing both 'bms' and 'altitude' in the filename
file_pattern = '*bms*altitude*.csv'

# ================================
# Setup Logging
# ================================

# Configure logging to write to a file with INFO level
logging.basicConfig(
    filename=os.path.join(directory, 'file_processing.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ================================
# File Selection
# ================================

# Retrieve all CSV files in the directory matching the pattern
all_files = glob.glob(os.path.join(directory, file_pattern))

# Check if any files are found
if not all_files:
    message = f"No CSV files found matching pattern '{file_pattern}' in directory: {directory}"
    print(message)
    logging.info(message)
    exit()

print(f"Found {len(all_files)} files matching the pattern '{file_pattern}'.")
logging.info(f"Found {len(all_files)} files matching the pattern '{file_pattern}'.")

# ================================
# Processing Files
# ================================

for file_path in tqdm(all_files, desc="Processing files"):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if 'altitude' column exists
        if 'altitude' not in df.columns:
            message = f"'altitude' column not found in {file_path}. Skipping file."
            print(message)
            logging.warning(message)
            continue

        # Calculate the number of unique altitude values
        unique_altitudes = df['altitude'].nunique()

        if unique_altitudes <= 3:
            basename = os.path.basename(file_path)
            new_basename = basename.replace('bms_altitude', 'bms')

            if 'bms_altitude' not in basename:
                logging.warning(f"'bms_altitude' not found in filename {basename}. Skipping rename.")
                continue

            new_file_path = os.path.join(directory, new_basename)
            df = df.drop(columns=['altitude'])
            if 'lat' and 'lng' in df.columns:
                df= df.drop(columns=['lat', 'lng'])
            if 'altitude' in df.columns:
                df= df.drop(columns=['altitude'])
            df.to_csv(new_file_path, index=False)
            logging.info(f"Renamed and updated file: {basename} -> {new_basename}")

            # Remove the original file
            os.remove(file_path)
            message = f"Removed original file: {basename}"
            print(message)
            logging.info(message)

        else:
            message = f"File '{os.path.basename(file_path)}' has {unique_altitudes} unique altitude values. No action taken."
            print(message)
            logging.info(message)

    except Exception as e:
        message = f"Error processing file '{file_path}': {e}"
        print(message)
        logging.error(message)
