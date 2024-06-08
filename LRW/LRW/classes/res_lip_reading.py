import torchvision.models as models
from torch import nn
from classes.cnn import CNN
from classes.lstm import LSTM
from classes.transformer import Transformer
from classes.video_cnn import ResNet, BasicBlock
import torch.nn.functional as F
import torch

class LipReadingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
        self.bn = nn.BatchNorm3d(num_features=64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        #resnet18 = models.resnet18(pretrained=True)
        #self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])

        self.lstm = nn.LSTM(input_size=512, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(1024 * 2, 500)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # print("Input shape:", x.shape)
        batch_size, seq_len, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4) # (batch, channels, time, height, width)
        x = self.bn(self.conv3d(x))
        x = self.pool(self.relu(x))
        # print("After Conv3d and Pooling shape:", x.shape)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, time, channels, height, width)
        # print("After Permute shape:", x.shape)

        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet18(x)

        x = x.view(batch_size, -1, 512) 
        # [16, 29, 512]
        # print("After View shape:", x.shape)
        x, _ = self.lstm(x)
        x = self.fc(self.dropout(x)).mean(1)
        # x = x.view(x.size(0), x.size(1), -1)  # (batch, time, features)
        # x, _ = self.lstm(x)
        # x = self.fc(x)
        return x