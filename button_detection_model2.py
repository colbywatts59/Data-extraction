import torch.nn as nn
import torch.nn.functional as F

window_len = 200

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x).squeeze(-1)  # (batch, 256)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CNNTransformerClassifier(nn.Module):
    def __init__(self, seq_len=200, num_classes=4, d_model=128, nhead=4, num_layers=2):
        super().__init__()

        # Local pattern extraction
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, d_model, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(d_model)

        # Transformer encoder expects [seq_len, batch_size, d_model]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global pooling + classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [batch_size, 1, seq_len]
        x = F.relu(self.bn1(self.conv1(x)))     # -> [B, 64, L]
        x = F.relu(self.bn2(self.conv2(x)))     # -> [B, d_model, L]

        x = x.permute(0, 2, 1)                  # -> [B, L, d_model]
        x = self.transformer_encoder(x)         # -> [B, L, d_model]

        x = x.permute(0, 2, 1)                  # -> [B, d_model, L]
        x = self.global_pool(x).squeeze(-1)     # -> [B, d_model]

        x = F.relu(self.fc1(x))
        return self.fc2(x)                      # -> [B, num_classes]