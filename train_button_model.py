# Script to train a CNN model to classify button presses based on peak data

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import button_detection_model2 as bdm

# Path to directory containing CSV files
path = "augmented_peaks"

def load_data():
    csv_files = sorted([f for f in os.listdir(path) if f.endswith(".csv")])

    all_peaks = []
    labels = []
    
    for file in csv_files:
        file_path = os.path.join(path, file)

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

    #print(all_peaks)
    #print(labels)

    print("Example Peak:", all_peaks[0])
    print("Example Label:", labels[0])

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    print("Encoded Labels:", encoded_labels)
    print("Label Mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

    # Convert to tensors and return
    return torch.tensor(all_peaks, dtype=torch.float32), torch.tensor(encoded_labels, dtype=torch.long)

        

X, y = load_data()

X = X.unsqueeze(1) # Add a channel dimension: [batch_size, 1, signal_length]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Get weights 
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())

# Convert to a PyTorch tensor
class_weights = torch.tensor(class_weights, dtype=torch.float32)

print("Class Weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)

model = bdm.CNNTransformerClassifier()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

print("Train class distribution:", torch.bincount(y_train))
print("Val class distribution:", torch.bincount(y_test))


num_epochs = 200
        
# Early stopping parameters
patience = 20
best_val_accuracy = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
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

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}, Val Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_since_improvement = 0
        torch.save(model.state_dict(), "bdm_CNN_Transformer_augmented.pt")  # Save best model
    else:
        epochs_since_improvement += 1

    if epochs_since_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
