import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CNN_LSTM, self).__init__()

        # CNN over time axis
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)     # (B, F, T)
        x = self.cnn(x)            # (B, 64, T')
        x = x.permute(0, 2, 1)     # (B, T', 64)

        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out
