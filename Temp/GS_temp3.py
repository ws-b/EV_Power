import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from GS_vehicle_dict import vehicle_dict  # Ensure this module is available

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Progress Bar
from tqdm import tqdm

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

# -------------------- Configuration -------------------- #

# Directory containing the CSV files
directory = r'D:\SamsungSTF\Processed_Data\TripByTrip'

# Select the vehicle
selected_car = 'EV6'
device_ids = vehicle_dict.get(selected_car, [])

if not device_ids:
    print(f"No device IDs found for the selected vehicle: {selected_car}")
    exit()

# Find all relevant CSV files containing 'bms' and 'altitude'
all_files = glob.glob(os.path.join(directory, '*bms*altitude*.csv'))

# Filter files that match the device IDs
files = [file for file in all_files if any(device_id in os.path.basename(file) for device_id in device_ids)]

if not files:
    print(f"No files found for the selected vehicle: {selected_car}")
    exit()

print(f"Processing {len(files)} files for vehicle: {selected_car}")

# Lists to store results
ml_results = []
residual_ratio_data = []
constant_altitude_files = []

# -------------------- Data Processing -------------------- #

for file in tqdm(files[:1000], desc="Processing files", unit="file"):
    try:
        # Read CSV file
        df = pd.read_csv(file, parse_dates=['time'])

        # Skip files with <=3 unique altitude values
        if df['altitude'].nunique() <= 3:
            constant_altitude_files.append(os.path.basename(file))
            print(f"Skipping {os.path.basename(file)}: 'altitude' has {df['altitude'].nunique()} unique values.")
            continue

        # Check for required columns
        required_columns = ['Power_phys', 'Power_data', 'altitude', 'speed', 'acceleration']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns in {file}: {missing_columns}")
            continue

        # Sort by time
        df.sort_values('time', inplace=True)

        # Set 'time' as index
        df.set_index('time', inplace=True)

        # Resample to 2-second intervals
        full_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='2S')
        df = df.reindex(full_time_index)

        # Interpolate 'altitude' based on time
        df['altitude'] = df['altitude'].interpolate(method='time')

        # Calculate Residual_Ratio
        df['Residual_Ratio'] = (df['Power_data'] - df['Power_phys']) / df['Power_phys']

        # Calculate altitude difference
        df['altitude_diff'] = df['altitude'].diff()

        # Label slopes
        def label_slope(x):
            if x > 0:
                return 'uphill'
            elif x < 0:
                return 'downhill'
            else:
                return 'flat'

        df['slope'] = df['altitude_diff'].apply(label_slope)

        # Exclude 'flat' segments
        data_ml = df[df['slope'] != 'flat'].dropna()

        if data_ml.empty:
            print(f"No valid data in {file} after excluding 'flat' segments.")
            continue

        # Prepare ML data and collect necessary columns for analysis
        X = data_ml[['Residual_Ratio', 'speed', 'acceleration']]
        y = data_ml['slope'].map({'uphill': 1, 'downhill': 0})  # 1: Uphill, 0: Downhill

        # Collect Residual_Ratio, speed, acceleration, and slope for Visualization
        residual_ratio_data.append(pd.DataFrame({
            'Residual_Ratio': X['Residual_Ratio'],
            'speed': X['speed'],
            'acceleration': X['acceleration'],
            'slope': y
        }))

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        ml_results.append({
            'file': file,
            'classification_report': report
        })

        print(f"Completed processing {os.path.basename(file)}")

    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

# -------------------- Results Summary -------------------- #

print("\n=== Machine Learning Classification Reports ===")
for result in ml_results:
    print(f"\nClassification Report for {os.path.basename(result['file'])}:")
    report_df = pd.DataFrame(result['classification_report']).transpose()
    print(report_df)

if constant_altitude_files:
    print("\n=== Files with Constant Altitude ===")
    for fname in constant_altitude_files:
        print(f"- {fname}")

# -------------------- Residual_Ratio Visualization -------------------- #

# Combine all Residual_Ratio data
all_residual_data = pd.concat(residual_ratio_data, ignore_index=True)

# Map slope to readable labels
all_residual_data['Slope_Label'] = all_residual_data['slope'].map({0: 'Downhill', 1: 'Uphill'})

# -------------------- Descriptive Statistics -------------------- #

print("\n=== Residual_Ratio Statistics by Slope ===")
print(all_residual_data.groupby('Slope_Label')['Residual_Ratio'].describe())

# -------------------- Boxplot Plotting -------------------- #
# Filter to remove extreme outliers (you can adjust the range based on your data)
filtered_data = all_residual_data[
    (all_residual_data['Residual_Ratio'] > -500) & (all_residual_data['Residual_Ratio'] < 500)
]

# Replot the boxplot after filtering out the outliers
plt.figure(figsize=(8, 6))
sns.boxplot(
    x='Slope_Label',
    y='Residual_Ratio',
    data=filtered_data,
    palette={'Downhill': 'blue', 'Uphill': 'red'}
)
plt.title('Residual_Ratio Boxplot by Slope (Filtered)')
plt.xlabel('Slope')
plt.ylabel('Residual Ratio')
plt.tight_layout()
plt.show()

# -------------------- Correlation Heatmap -------------------- #
# Compute the correlation matrix including Residual_Ratio, speed, acceleration, and slope
correlation_matrix = all_residual_data[['Residual_Ratio', 'speed', 'acceleration', 'slope']].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()
