import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Vehicle dict
vehicle_dict = {
    'NiroEV': ['01241228149', '01241228151', '01241228153', '01241228154', '01241228155'],
    'Ionic5': ['01241227999', '01241228003', '01241228005', '01241228007', '01241228009', '01241228014',
               '01241228016', '01241228020', '01241228021', '01241228024', '01241228025', '01241228026', '01241228030',
               '01241228037', '01241228044', '01241228046', '01241228047', '01241248780', '01241248782',
               '01241248790', '01241248811', '01241248815', '01241248817', '01241248820', '01241248827',
               '01241364543', '01241364560', '01241364570', '01241364581', '01241592867', '01241592868',
               '01241592878', '01241592896', '01241592907', '01241597801', '01241597802', '01241248919',
               '01241321944'],
    'Ionic6': ['01241248713', '01241592904', '01241597763', '01241597804'],
    'KonaEV': ['01241228102', '01241228122', '01241228123', '01241228156', '01241228197', '01241228203',
               '01241228204', '01241248726', '01241248727', '01241364621', '01241124056'],
    'EV6': ['01241225206', '01241228048', '01241228049', '01241228050', '01241228051', '01241228053',
            '01241228054', '01241228055', '01241228057', '01241228059', '01241228073', '01241228075',
            '01241228076', '01241228082', '01241228084', '01241228085', '01241228086', '01241228087',
            '01241228090', '01241228091', '01241228092', '01241228094', '01241228095', '01241228097',
            '01241228098', '01241228099', '01241228103', '01241228104', '01241228106', '01241228107',
            '01241228114', '01241228124', '01241228132', '01241228134', '01241248679', '01241248818',
            '01241248831', '01241248833', '01241248842', '01241248843', '01241248850', '01241248860',
            '01241248876', '01241248877', '01241248882', '01241248891', '01241248892', '01241248900',
            '01241248903', '01241248908', '01241248909', '01241248912', '01241248913', '01241248921', '01241248924',
            '01241248926', '01241248927', '01241248929', '01241248932', '01241248933', '01241248934',
            '01241321943', '01241321947', '01241364554', '01241364575', '01241364592', '01241364627',
            '01241364638', '01241364714', '01241248928'],
    'GV60': ['01241228108', '01241228130', '01241228131', '01241228136', '01241228137', '01241228138'],
    'Porter2EV': ['01241228144', '01241228160', '01241228177', '01241228188', '01241228192', '01241228171'],
    'Bongo3EV': ['01241228162', '01241228179', '01241248642', '01241248723', '01241248829']
}

# Directory containing CSV files
directory = r"D:\SamsungSTF\Processed_Data\TripByTrip"


def process_file(file_path):
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)
        device_id = os.path.basename(file_path).split('-')[1]
        # Convert speed from m/s to km/h
        data['speed'] = data['speed'] * 3.6
        return device_id, data['speed'], data['acceleration']
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None, None


def sample_grid(speed, acceleration, grid_size=100, max_per_grid=30):
    grid = {}
    min_speed, max_speed = speed.min(), speed.max()
    min_accel, max_accel = acceleration.min(), acceleration.max()

    speed_bins = np.linspace(min_speed, max_speed, grid_size + 1)
    accel_bins = np.linspace(min_accel, max_accel, grid_size + 1)

    for s, a in zip(speed, acceleration):
        s_bin = np.digitize(s, speed_bins) - 1
        a_bin = np.digitize(a, accel_bins) - 1
        grid_key = (s_bin, a_bin)
        if grid_key not in grid:
            grid[grid_key] = []
        if len(grid[grid_key]) < max_per_grid:
            grid[grid_key].append((s, a))

    sampled_speeds = []
    sampled_accelerations = []
    for points in grid.values():
        sampled_speeds.extend([p[0] for p in points])
        sampled_accelerations.extend([p[1] for p in points])

    return sampled_speeds, sampled_accelerations


# Iterate through each vehicle type in the vehicle_dict
for vehicle_type, device_ids in vehicle_dict.items():
    all_speeds = []
    all_accelerations = []

    # Get the list of relevant files
    relevant_files = [os.path.join(directory, f) for f in os.listdir(directory) if
                      f.endswith(".csv") and any(device_id in f for device_id in device_ids)]

    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, relevant_files)

        for device_id, speeds, accelerations in results:
            if speeds is not None and accelerations is not None:
                all_speeds.extend(speeds)
                all_accelerations.extend(accelerations)

    # Convert to numpy arrays for easy handling
    all_speeds = np.array(all_speeds)
    all_accelerations = np.array(all_accelerations)

    # Sample the data in the grids
    sampled_speeds, sampled_accelerations = sample_grid(all_speeds, all_accelerations)

    # Create a scatter plot for the current vehicle type
    plt.figure(figsize=(10, 6))
    plt.scatter(sampled_speeds, sampled_accelerations, alpha=0.5)
    plt.title(f'Speed vs Acceleration for {vehicle_type}')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Acceleration (m/s²)')
    plt.xlim(0, 200)
    plt.ylim(-15, 9)
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(os.path.join(directory, f'{vehicle_type}_speed_vs_acceleration.png'))
    plt.close()

print("Plots generated and saved successfully.")