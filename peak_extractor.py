# Script to extract peaks from the CSV file using the trained peak_extractor.pt model
# First the CNN model is used to predict the peaks in the data
# Then the predicted peaks are smoothed using a moving average to get rid of noise
# Finally, the smoothed peaks are clustered using KMeans clustering to determine where each peak starts and ends
# The output is a CSV file named after the button with the start and end of each peak

# If the CSV file already exists, it will append to the file so be sure not to run the script multiple times

# Name of the button
button_name = "bottom right"

# Path to the folder where button_name is located
path = "test_data/"

# Path to the folder where the output CSV file will be saved
output_path = "test_data/individual_peaks"

num_peaks_in_trace = 3

import os
import json
import pandas as pd
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans

import peak_detection_model as pdm


full_path = os.path.join(path, button_name)

# Load the model
window_size = 40  # Set the window size
model = pdm.PeakCNN(window_size=window_size)
model.load_state_dict(torch.load("peak_extractor.pt"))
model.eval()

def preprocess_csv_for_inference_cnn(csv_path, window_size=40):
    df = pd.read_csv(csv_path, skiprows=21)
    df = df.rename(columns={df.columns[2]: 'Voltage'})
    df = df.drop(columns=[df.columns[3], df.columns[4]], errors='ignore')
    df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
    df = df.dropna(subset=['Voltage'])
    df['Moving_Avg_Voltage'] = df['Voltage'].rolling(window=30).mean()
    df = df.dropna().reset_index(drop=True)

    signal = df['Moving_Avg_Voltage'].values

    # Normalize the signal
    signal = (signal - np.mean(signal)) / np.std(signal)
    X_test = []

    # Generate windows for CNN input (ensuring the correct input shape)
    for i in range(window_size, len(signal) - window_size):
        window = signal[i - window_size:i + window_size + 1]
        X_test.append(window)

    # Load train_mean and train_std
    with open("normalization_stats.json", "r") as f:
        stats = json.load(f)
        train_mean = stats["mean"]
        train_std = stats["std"]

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    X_test = (X_test - train_mean) / train_std

    # Add channel dimension so it's [batch_size, 1, signal_length]
    X_test = X_test.unsqueeze(1)
    
    return X_test, df.iloc[window_size:-window_size].reset_index(drop=True)
  

# Get all CSV files in the folder
csv_files = sorted([f for f in os.listdir(full_path) if f.endswith(".csv")])

# List to store all the peaks 
all_peaks = []

# Loop through eac h CSV file
for file in csv_files:
    file_path = os.path.join(full_path, file)
    X_test, df = preprocess_csv_for_inference_cnn(file_path, window_size=window_size)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_test)
        predictions = predictions.sigmoid()
        predicted_labels = (predictions > 0.7).int()

    # Smooth predicted labels with a moving average to get rid of noise
    smoothed_labels = uniform_filter1d(predicted_labels.numpy(), size=4)
    smoothed_labels = (smoothed_labels > 0.5).astype(int)

    peak_indices = np.where(smoothed_labels == 1)[0]

    peak_positions = np.array(peak_indices).reshape(-1, 1)

    print(f"File: {file}, Number of peaks detected: {len(peak_indices)}")

    # Use KMeans clustering to determine where each peak starts and ends
    kmeans = KMeans(n_clusters=num_peaks_in_trace, random_state=42).fit(peak_positions)  
    labels = kmeans.labels_

    peak_ranges = []
    # Get the start and end of each peak
    for cluster in range(num_peaks_in_trace):
        cluster_indices = peak_indices[labels == cluster].flatten()
        start_index = cluster_indices[0]
        end_index = cluster_indices[-1]
        start_peak = df['Sample Number'].iloc[start_index]
        end_peak = df['Sample Number'].iloc[end_index]
        peak_ranges.append((start_peak, end_peak))
        
    # Sort the peak ranges by start index
    peak_ranges = sorted(peak_ranges, key=lambda x: x[0])

    # Print the peaks for each file
    print(f"File: {file}")
    for start, end in peak_ranges:
        print(f"    Peak range: {start} to {end}")
        peak = df['Voltage'].iloc[start:end + 1]
        peak_row = ",".join(map(str, peak.values))
        all_peaks.append({"File": {file}, "Peak Range": f"{start}-{end}", "Peak Data": peak_row})

# Save output_df to csv
output_df = pd.DataFrame(all_peaks)

output_file = f"{output_path}/{button_name}_peaks.csv"

if os.path.exists(output_file):
    # Append to file
    output_df.to_csv(output_file, mode='a', header=False, index=False)
else:
    # Create new csv
    output_df.to_csv(output_file, index=False)