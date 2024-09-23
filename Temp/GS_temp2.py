import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from GS_vehicle_dict import vehicle_dict
# -------------------- 설정 --------------------

# CSV 파일들이 있는 디렉토리 경로를 설정합니다.
directory = r'D:\SamsungSTF\Processed_Data\TripByTrip'
selected_car = 'EV6'

device_ids = vehicle_dict.get(selected_car, [])

if not device_ids:
    print(f"선택한 차량에 대한 디바이스 ID를 찾을 수 없습니다: {selected_car}")
    exit()

# 디렉토리에서 'bms'와 'altitude'가 포함된 모든 CSV 파일을 가져옵니다.
all_files = glob.glob(os.path.join(directory, '*bms*altitude*09*.csv'))

# 단말기 번호가 파일명에 포함된 파일만 선택합니다.
files = [file for file in all_files if any(device_id in os.path.basename(file) for device_id in device_ids)]

if not files:
    print(f"선택한 차량에 대한 파일을 찾을 수 없습니다: {selected_car}")
    exit()

print(f"처리할 파일 수: {len(files)} (차량: {selected_car})")

# -------------------- 데이터 처리 함수 --------------------
def process_file(file_path):
    # Read CSV
    try:
        df = pd.read_csv(file_path, parse_dates=['time'])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Check for required columns
    required_columns = ['time', 'Power_phys', 'Power_data', 'altitude']
    if not all(col in df.columns for col in required_columns):
        print(f"Missing required columns in {file_path}. Skipping this file.")
        return None

    # Sort by time to ensure proper interpolation
    df = df.sort_values('time').reset_index(drop=True)

    # Calculate Power Ratio
    df['Power_Ratio'] = df['Power_data'] / df['Power_phys']

    # Interpolate 'altitude' based on time
    df.set_index('time', inplace=True)
    df['altitude_interpolated'] = df['altitude'].interpolate(method='time')

    # Classification based on interpolated altitude
    df['altitude_diff'] = df['altitude_interpolated'].diff()
    df['altitude_slope'] = df['altitude_diff'].apply(
        lambda x: 'Uphill' if x > 0 else ('Downhill' if x < 0 else 'Flat')
    )

    # Reset index to have 'time' as a column again
    df.reset_index(inplace=True)

    # Add filename column for reference
    df['filename'] = os.path.basename(file_path)

    return df


# -------------------- Process All Files and Aggregate Data --------------------

processed_dfs = []

for file in files:
    print(f"Processing file: {file}")
    processed_df = process_file(file)
    if processed_df is not None:
        processed_dfs.append(processed_df)

if not processed_dfs:
    print("No data processed. Exiting.")
    exit()

# Concatenate all processed DataFrames
combined_df = pd.concat(processed_dfs, ignore_index=True)
print(f"Combined DataFrame shape: {combined_df.shape}")

# -------------------- Prepare Data --------------------

# Define features and labels
features = combined_df[['Power_Ratio', 'altitude_interpolated']]
labels = combined_df['altitude_slope']

# Handle missing values
features = features.fillna(method='ffill').fillna(method='bfill')
labels = labels.fillna('Flat')

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Feature Scaling (optional)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# -------------------- Split Data into Training and Validation Sets --------------------

X_train, X_val, y_train, y_val = train_test_split(
    features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# -------------------- Model Training --------------------

# Initialize Random Forest model
rf_classifier = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=None,    # Maximum depth of the tree
    random_state=42
)

# Train the model
rf_classifier.fit(X_train, y_train)
print("Random Forest model training completed.")

# -------------------- Model Evaluation --------------------

# Predict on validation set
y_pred = rf_classifier.predict(X_val)

# Accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Classification Report
print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# -------------------- Feature Importance --------------------

importances = rf_classifier.feature_importances_
feature_names = ['Residual Ratio', 'altitude_interpolated']
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()