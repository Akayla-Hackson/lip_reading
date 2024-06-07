import torchvision.models as models
from torch import nn
from classes.cnn import CNN
from classes.lstm import LSTM
from classes.transformer import Transformer
import torch.nn.functional as F
import torch


class LipReadingModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.lstm = nn.LSTM(input_size=64 * 48 * 48, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, len(self.vocab))
        

    def forward(self, x):
        # print("Input shape:", x.shape)
        batch_size, seq_len, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4) # (batch, channels, time, height, width)
        x = self.pool(self.relu(self.conv3d(x)))
        # print("After Conv3d and Pooling shape:", x.shape)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, time, channels, height, width)
        # print("After Permute shape:", x.shape)
        x = x.view(x.size(0), x.size(1), -1)  # (batch, time, features)
        # print("After View shape:", x.shape)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x