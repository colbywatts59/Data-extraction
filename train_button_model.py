# Script to train a CNN model to classify button presses based on peak data

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import argparse
import json

import button_detection_model2 as bdm

# Path to directory containing CSV files (can be overridden by --data-dir)
path = "augmented_peaks"

def load_data(button_filters=None):
    csv_files = sorted([f for f in os.listdir(path) if f.endswith(".csv")])

    # Normalize filters for case-insensitive matching
    if button_filters is not None:
        filter_set = set([b.lower() for b in button_filters])
    else:
        filter_set = None

    all_peaks = []
    labels = []
    
    for file in csv_files:
        file_path = os.path.join(path, file)

        button_name = file.split("_")[0]
        button_name_norm = button_name.lower()

        # Apply filter if provided
        if filter_set is not None and button_name_norm not in filter_set:
            continue

        df = pd.read_csv(file_path)

        peak = df['Peak Data'].apply(lambda x: np.array(x.split(",")).astype(float))

        for single_peak in peak:
            single_peak = (single_peak - single_peak.mean()) / (single_peak.std() + 1e-6)
            if len(single_peak) < bdm.window_len:
                single_peak = np.pad(single_peak, (0, bdm.window_len - len(single_peak)))
            elif len(single_peak) > bdm.window_len:
                single_peak = single_peak[:bdm.window_len]

            all_peaks.append(single_peak)
            labels.append(button_name)

    if len(all_peaks) == 0:
        raise ValueError("No data found after applying button filters. Check --buttons values and data-dir.")

    print("Example Peak:", all_peaks[0])
    print("Example Label:", labels[0])

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    print("Encoded Labels:", encoded_labels)
    print("Label Mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

    # Convert to tensors and return
    return (
        torch.tensor(all_peaks, dtype=torch.float32),
        torch.tensor(encoded_labels, dtype=torch.long),
        label_encoder.classes_.tolist(),
    )

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train button classifier with optional button filtering")
    parser.add_argument("--buttons", nargs="+", help="Buttons to include, e.g., 'top left' 'bottom right'", default=None)
    parser.add_argument("--data-dir", type=str, default="augmented_peaks", help="Directory with augmented CSVs")
    parser.add_argument("--model-out", type=str, default="bdm_CNN_augmented3.pt", help="Path to save model weights")
    args = parser.parse_args()

    # Override data path
    path = args.data_dir

    if args.buttons is not None:
        print("Training with buttons:", args.buttons)
    else:
        print("Training with all available buttons in", path)

    X, y, classes = load_data(button_filters=args.buttons)

    # Save classes alongside the model
    classes_path = f"{args.model_out}.classes.json"
    with open(classes_path, "w") as f:
        json.dump(classes, f)

    X = X.unsqueeze(1) # Add a channel dimension: [batch_size, 1, signal_length]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)

    # Use a weighted sampler to address class imbalance
    class_counts = np.bincount(y_train.numpy())
    class_weights_for_sampler = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights_for_sampler[y_train.numpy()]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Get weights 
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())

    # Convert to a PyTorch tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print("Class Weights:", class_weights)

    # Set number of classes dynamically from selected labels
    num_classes = int(len(torch.unique(y)))
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = bdm.CNNTransformerClassifier(num_classes=num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("Train class distribution:", torch.bincount(y_train))
    print("Val class distribution:", torch.bincount(y_test))


    num_epochs = 200
            
    # Early stopping parameters
    patience = 20
    best_val_accuracy = 0
    epochs_since_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()               # NOTE WHAT IF I CHANGE SIZE OF INPUTS NOT MAKE IT 200
        
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                preds = model(inputs)
                val_preds.append(preds)
                val_targets.append(targets)
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_accuracy = (val_preds.argmax(dim=1) == val_targets).float().mean().item()

       
        

        if val_accuracy > best_val_accuracy:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}, Val Accuracy: {val_accuracy:.4f} New Best!")
            best_val_accuracy = val_accuracy
            epochs_since_improvement = 0
            torch.save(model.state_dict(), args.model_out)  # Save best model
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}, Val Accuracy: {val_accuracy:.4f}")
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
