import torchvision.models as models
from torch import nn
from classes.cnn import CNN
from classes.lstm import LSTM
from classes.transformer import Transformer
import torch.nn.functional as F
import torch


class LipReadingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.relu = nn.ReLU()
        self.maxpool3d = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # ResNet-18 model
        resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(256*2, 500)
        
    def forward(self, x):
        # print("Input shape:", x.shape)
        batch_size, seq_len, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4) # (batch, channels, time, height, width)
        x = self.pool(self.relu(self.conv3d(x)))
        batch_size, channels, depth, height, width = x.size()
        x = x.view(batch_size * depth, channels, height, width)
        x = self.resnet18(x)
        x = x.view(batch_size, depth, -1)
        
        # Apply LSTM layer
        x, _ = self.lstm(x)
        
        # Apply Fully Connected layer to the last LSTM output
        x = self.fc(x[:, -1, :])
        return x