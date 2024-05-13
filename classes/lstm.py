import torchvision.models as models
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)
