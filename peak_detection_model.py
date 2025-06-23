# Defines a pytorch neural network model for peak detection

import torch.nn as nn

class PeakCNN(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        input_length = 2 * window_size + 1

        # Two maxpools of kernel size 2 will divide the length by 4
        conv_output_length = input_length // 4

        # Two convolutional layers with ReLU activations and max pooling
        # followed by a dropout layer and a linear layer
        # The final output is a single value (0 or 1)
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(32 * conv_output_length, 1)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.net(x).squeeze(1)