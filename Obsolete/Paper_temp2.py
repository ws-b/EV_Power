import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the folder path where CSV files are located
folder_path = r"D:\SamsungSTF\Data\Cycle\HW_GS"

# Define color options for different files
colors = ['green', 'orange']  # New color choices
file_count = 0  # Initialize file counter

plt.figure(figsize=(11, 4))

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the required columns exist in the CSV
        if 'time' in df.columns and 'Power_data' in df.columns:
            # Convert the 'time' column to datetime
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
            df['elapsed_time'] = df['time'] - df['time'].iloc[0]  # Calculate elapsed time
            df['elapsed_time'] = df['elapsed_time'].dt.total_seconds()

            # Plot Power_data over elapsed time, dividing Power_data by 1000 for kW
            plt.plot(df['elapsed_time'], df['Power_data'] / 1000,
                     label=f'Power(kW) - Highway Cycle {file_count + 1}', color=colors[file_count])

            file_count += 1  # Increment file counter to switch color

# Label the axes
plt.xlabel('Elapsed Time (seconds)')
plt.ylabel('Power (kW)')

# Add the legend
plt.legend(['Highway Cycle 1', 'Highway Cycle 2'], loc='upper right')

# Remove grid
plt.grid(False)
plt.tight_layout()
# Display the plot
plt.show()
