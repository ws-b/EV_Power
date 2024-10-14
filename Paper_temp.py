import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the folder path where CSV files are located
folder_path = r"D:\SamsungSTF\Data\Cycle\CITY_GS"

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the required columns exist in the CSV
        if 'time' in df.columns and 'speed' in df.columns and 'acceleration' in df.columns:
            # Convert the 'time' column to datetime and calculate elapsed time in seconds
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
            df['elapsed_time'] = df['time'] - df['time'].iloc[0]  # Calculate elapsed time
            df['elapsed_time'] = df['elapsed_time'].dt.total_seconds()

            plt.figure(figsize=(12,4), dpi = 300)

            # Create a plot with speed on the primary y-axis
            fig, ax1 = plt.subplots(figsize=(12, 4), dpi = 300)

            ax1.set_xlabel('Elapsed Time (seconds)')
            ax1.set_ylabel('Speed', color='tab:blue')
            ax1.plot(df['elapsed_time'], df['speed'], label='Speed', color='tab:blue', alpha = 0.7)
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            # Create a second y-axis for acceleration
            ax2 = ax1.twinx()
            ax2.set_ylabel('Acceleration', color='tab:red')
            ax2.plot(df['elapsed_time'], df['acceleration'], label='Acceleration', color='tab:red', alpha = 0.7)
            ax2.tick_params(axis='y', labelcolor='tab:red')

            # Remove grid
            ax1.grid(False)
            ax2.grid(False)
            plt.tight_layout()
            # Show the plot
            plt.show()
