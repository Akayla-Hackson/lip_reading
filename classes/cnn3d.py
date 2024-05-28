import torch.nn as nn
import torch

class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.global_avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1)) # Pool only spatial dimensions

    def forward(self, x):
        batch_size, channels, seq_len,h, w = x.shape
        x = self.pool1(self.relu(self.conv1(x)))
        # print("x after 1st pool & relu shape:", x.shape)
        x = self.pool2(self.relu(self.conv2(x)))
        # print("x after 2nd pool & relu shape:", x.shape)
        x = self.global_avg_pool(x)  # Output shape: [batch_size, num_channels, seq_len, 1, 1]
        # print("x after global_avg_pool shape:", x.shape)
        x = x.squeeze(-1).squeeze(-1)  # Shape: [batch_size, num_channels, seq_len]
        # print("x after squeeze shape:", x.shape)
        x=x.permute(0,2,1)
        # print("x after permute shape:", x.shape)
        return x
