import torchvision.models as models
from torch import nn
from classes.cnn import CNN
from classes.lstm import LSTM
from classes.hw_transformer import LipReadingTransformer
import torch.nn.functional as F
import torch

class HwLipReadingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN() 
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.transformer = LipReadingTransformer(feature_size=256, num_heads=8, num_layers=6)

    def forward(self, x, tgt_tokens, training):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)

        cnn_out = self.cnn(x)    # Expected shape: (batch_size, updated num channels, updated height, updated width)
        cnn_out = cnn_out.view(batch_size, seq_len, -1) 

        lstm_out, _ = self.lstm(cnn_out)   # Expected shape: (batch_size, sequence_length (num_frames), features)

        if training:
            output = self.transformer(lstm_out, tgt_tokens) # Expected shape: (batch_size, sequence_length, vocab size)
        else:
            output = self.transformer.sample(lstm_out)
        
        return output