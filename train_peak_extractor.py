# Program to train a CNN model for peak detection in voltage signals

import torch
import numpy as np
import pandas as pd
import os
import json
from torch.utils.data import DataLoader, TensorDataset, random_split

import peak_detection_model as pdm

path_to_buttons = "train_data"

button_names = ["top left", "bottom right"]


# Function to load labeled data from JSON files and corresponding CSV files
def load_labeled_data(labeled_jsons_dir, window_size=40):
    X = []
    y = []

    for button_name in os.listdir(labeled_jsons_dir):
        if not button_name in button_names:
            continue
        subdir_json_path = os.path.join(labeled_jsons_dir, button_name)
        print(f"subdir path: {subdir_json_path}")
        # Skip any non-directories
        if not os.path.isdir(subdir_json_path):
            continue

        for trace in os.listdir(subdir_json_path):
            # Skip any non json files
            if not trace.endswith(".json"):
                continue

            print(f"Processing {button_name} - {trace}")
            # Open the json file and hte corresponding csv file
            json_file = os.path.join(subdir_json_path, trace)
            csv_file = os.path.join(path_to_buttons +  "/" + button_name, os.path.splitext(trace)[0] + ".csv")
            print(f"json_file: {json_file}")
            print(f"csv_file: {csv_file}")

            if not os.path.exists(json_file):
                print(f"json file not found: {json_file}")
                continue

            if not os.path.exists(csv_file):
                print(f"csv file not found: {csv_file}")
                continue

            # Load and process CSV
            df = pd.read_csv(csv_file, skiprows=21)
            df = df.rename(columns={df.columns[2]: 'Voltage'})
            df = df.drop(columns=[df.columns[3], df.columns[4]], errors='ignore')
            df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
            df = df.dropna(subset=['Voltage'])
            df['Moving_Avg_Voltage'] = df['Voltage'].rolling(window=30).mean()
            df = df.dropna().reset_index(drop=True)

            # Create an array of zeros for labels
            # 0 will indicate no peak, 1 will indicate a peak
            labels = np.zeros(len(df), dtype=int)
            with open(json_file) as f:
                peaks = json.load(f)
                for peak in peaks:
                    #print(f"Peak: {peak}")
                    start, end = peak['range']

                    # Update labels within peak range to 1
                    labels[(df['Sample Number'] >= start) & (df['Sample Number'] <= end)] = 1
                    print(f"Labels: {np.unique(labels, return_counts=True)}")

            signal = df['Moving_Avg_Voltage'].values

            # Normalize the signal
            signal = (signal - np.mean(signal)) / np.std(signal)


            # Create windows for CNN input
            for i in range(window_size, len(signal) - window_size):
                window = signal[i - window_size:i + window_size + 1]
                X.append(window) # Contains windows of the signal
                y.append(labels[i]) # Contains the labels for the window

    # Convert to tensors and return
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)


# Data loading
window_size = 40
X, y = load_labeled_data("labeled_jsons", window_size=window_size)

# Normalize data
mean = X.mean()
std = X.std()
X = (X - mean) / std

# Save train_mean and train_std (used in visualize_peak_extractor.py)
with open("normalization_stats.json", "w") as f:
    json.dump({"mean": mean.item(), "std": std.item()}, f)

# Add channel dimension for CNN: [batch_size, 1, signal_length]
X = X.unsqueeze(1)

# Verify the shape before passing into the DataLoader
print("X shape:", X.shape) 

# Dataset and loaders
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset)) # 80/20 train/test split
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  
val_loader = DataLoader(val_dataset, batch_size=64)  

model = pdm.PeakCNN(window_size=window_size)

# Handle class imbalance to avoid bias toward majority class
pos_weight = torch.tensor([(y == 0).sum() / (y == 1).sum()], dtype=torch.float32)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

probability_threshold = 0.7 # Threshold for logits to be considered a peak

num_epochs = 100

# Early stopping parameters
patience = 15
best_val_accuracy = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        preds = model(inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step() 
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_preds = torch.cat([model(inputs) for inputs, _ in val_loader])
        val_targets = torch.cat([targets for _, targets in val_loader])
        val_accuracy = ((val_preds > 0.5).float() == val_targets).float().mean().item()

    predicted_labels = (val_preds.sigmoid() > probability_threshold).int() # Convert logits to binary labels

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_since_improvement = 0
        torch.save(model.state_dict(), "peak_extractor.pt")
    else:
        epochs_since_improvement += 1

    if epochs_since_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break
    # print(f"Raw logits: {val_preds[:10]}")
    # print(f"Probabilities: {val_preds.sigmoid()[:10]}")
