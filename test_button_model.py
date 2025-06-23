# Script to test the button_detection_model CNN on test data
# Shows accuracy and classification report

import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import torch
from sklearn.metrics import classification_report, confusion_matrix
import json

import button_detection_model2 as bdm


# Path to individual peak data
path = "test_data/individual_peaks"


def load_data():
    csv_files = sorted([f for f in os.listdir(path) if f.endswith(".csv")])

    all_peaks = []
    labels = []
    
    for file in csv_files:
        file_path = os.path.join(path, file)

        print(f"Processing file {file_path}")

        button_name = file.split("_")[0]

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

    # print("Example Peak:", all_peaks[0])
    # print("Example Label:", labels[0])

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    with open("label_encoder_classes.json", "w") as f:
        json.dump(label_encoder.classes_.tolist(), f)

    # print("Encoded Labels:", encoded_labels)
    # print("Label Mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

    # For performance, convert to numpy array
    all_peaks = np.array(all_peaks)
    return torch.tensor(all_peaks, dtype=torch.float32), torch.tensor(encoded_labels, dtype=torch.long), label_encoder

# Load model
model = bdm.CNNTransformerClassifier()
model.load_state_dict(torch.load("bdm_CNN_Transformer_augmented.pt"))
model.eval()

# Load and preprocess test data
x_data, y_data, label_encoder = load_data()
x_data = x_data.unsqueeze(1)  # Add channel dim: [batch_size, 1, window_len]
test_dataset = TensorDataset(x_data, y_data) 

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inference
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader: 
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds)
        all_labels.append(labels)

# Combine all batches
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# Accuracy
accuracy = (all_preds == all_labels).float().mean().item()
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Report
all_class_indices = list(range(len(label_encoder.classes_)))

print("\nClassification Report:")
print(classification_report(
    all_labels.numpy(),
    all_preds.numpy(),
    labels=all_class_indices,
    target_names=label_encoder.classes_
))



print("\nConfusion Matrix:")
print(confusion_matrix(all_labels.numpy(), all_preds.numpy()))
