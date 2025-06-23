# This script is used to test the peak detection model on a CSV file
# It loads the model, preprocesses the CSV file, makes predictions, and visualizes the results

# NOTE: This is not the final method used for extracting peaks
#       This uses an algorithmic approach to find peaks but it is flawed 
#       See peak_extractor.py for the final method that uses k-means clustering

import torch
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import label
import numpy as np

import peak_detection_model as pdm



# Path to button folder
path = "test_data/bottom left"

# Name of CSV file
trace = "Trace 7.csv"

full_path = os.path.join(path, trace)

# Load the model
window_size = 40  # Set the window size
model = pdm.PeakCNN(window_size=window_size)
model.load_state_dict(torch.load("peak_extractor.pt"))
model.eval()

# Preprocessing for CNN input
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

    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_test = (X_test - train_mean) / train_std

    # Add channel dimension: [batch_size, 1, signal_length]
    X_test = X_test.unsqueeze(1) 
    
    return X_test, df.iloc[window_size:-window_size].reset_index(drop=True)

# Usage
X_test, metadata_df = preprocess_csv_for_inference_cnn(full_path)

# Make predictions
with torch.no_grad():
    preds = model(X_test)
    preds = preds.sigmoid()
    predicted_labels = (preds > 0.7).int()

# Smooth predicted labels with a moving average
smoothed_labels = uniform_filter1d(predicted_labels.numpy(), size=4)
smoothed_labels = (smoothed_labels > 0.5).astype(int)
metadata_df['Smoothed Label'] = smoothed_labels


# Label connected components in the smoothed labels
labeled_array, num_features = label(metadata_df['Smoothed Label'])

num_peaks = num_features
    
peak_ranges = []

current_start = None
current_end = None

for peak_label in range(1, num_features + 1):
    # Get the start and end indices for the current peak
    indices = (labeled_array == peak_label).nonzero()[0]
    start_index = indices[0]
    end_index = indices[-1]
    start_sample = metadata_df['Sample Number'].iloc[start_index]
    end_sample = metadata_df['Sample Number'].iloc[end_index]

    # Check if the current peak should be merged with the next one
    if current_start is None:
        current_start = start_sample
        current_end = end_sample
    elif current_end - current_start < 160: # This number is about how long each peak is 
        # Extend the current peak
        current_end = end_sample
        num_peaks -= 1 
    else:
        peak_ranges.append((current_start, current_end))
        current_start = start_sample
        current_end = end_sample

# Add the last peak
if current_start is not None:
    peak_ranges.append((current_start, current_end))
        

# Print the merged ranges
print("Merged Peak Ranges:")
for start, end in peak_ranges:
    print(f"Start: {start}, End: {end}")
        

# Visualize the signal and predicted peaks
plt.figure(figsize=(12, 4))
plt.plot(metadata_df['Sample Number'], metadata_df['Moving_Avg_Voltage'], label='Signal')
plt.scatter(metadata_df['Sample Number'][metadata_df['Smoothed Label'] == 1],
            metadata_df['Moving_Avg_Voltage'][metadata_df['Smoothed Label'] == 1],
            color='red', label='Predicted Peak')
plt.title(f"Predicted Peaks in {trace}")
plt.legend()
plt.show()
