import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the path to the data
win_folder_path = 'D:\Data\대학교 자료\켄텍 자료\삼성미래과제\한국에너지공과대학교_샘플데이터\ioniq5'
folder_path = os.path.normpath(win_folder_path)

# Get a list of all files in the folder with the .csv extension
file_lists = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
file_lists.sort()

# Split the files into training and test sets
train_files, test_files = train_test_split(file_lists, test_size=0.5, random_state=42)

# Function to process each file and return a and b
def process_file(file):
    # Read the data from the file into a pandas DataFrame
    data = pd.read_csv(os.path.join(folder_path, file))

    # Update: Get the actual usage from the 'Energy_VI' column
    data['actual_usage'] = data['Energy_VI']

    # Calculate the energy with speed considered
    X = (data['Energy'] * data['emobility_spd_m_per_s']).values.reshape(-1, 1)
    y = data['actual_usage'].values

    # Fit the data using Linear Regression
    model = LinearRegression()
    model.fit(X, y)

    return model.intercept_, model.coef_[0]

# Calculate a and b for each file in the training set and get their averages
a_values = []
b_values = []
for file in tqdm(train_files, desc="Processing training files"):
    a, b = process_file(file)
    a_values.append(a)
    b_values.append(b)

average_a = np.mean(a_values)
average_b = np.mean(b_values)

# Function to predict and plot
def predict_and_plot(file, a, b):
    # Read the data from the file into a pandas DataFrame
    data = pd.read_csv(os.path.join(folder_path, file))

    # Update: Get the actual usage from the 'Energy_VI' column
    data['actual_usage'] = data['Energy_VI']

    # Calculate the energy with speed considered
    X = (data['Energy'] * data['emobility_spd_m_per_s']).values.reshape(-1, 1)

    # Predict using a and b
    y_pred = a + b * X

    # Convert the 'time' column to total minutes
    data['time'] = pd.to_datetime(data['time'])
    data['time'] = data['time'] - data['time'].iloc[0]
    data['time_in_minutes'] = data['time'].dt.total_seconds() / 60
    time_in_minutes = data['time_in_minutes'].values

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_in_minutes, data['actual_usage'].values, label='Actual Usage', color='red')
    plt.plot(time_in_minutes, data['Energy'].values, label='Energy Before Fitting', color='blue')
    plt.plot(time_in_minutes, y_pred, label='Predicted Usage', color='green')
    plt.xlabel('Time (minutes)')
    plt.legend(loc='upper left')

    # Display the file name at the top right of the plot
    plt.annotate(f'File: {file}', xy=(1, 1), xycoords='axes fraction', fontsize=12, ha='right', va='top')

    plt.show()


# Update: Select a random subset of 10 test files
random_test_files = np.random.choice(test_files, 5)

# Predict and plot for each file in the test set
for file in tqdm(random_test_files, desc="Processing test files"):
    predict_and_plot(file, average_a, average_b)
