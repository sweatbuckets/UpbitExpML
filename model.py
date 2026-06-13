import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, per_step_feature, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=per_step_feature,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])
