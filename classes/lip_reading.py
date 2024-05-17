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
        self.cnn = CNN()
        self.lstm = LSTM(input_dim=512, hidden_dim=256, num_layers=1)  
        self.transformer = Transformer(feature_size=256, num_tokens=30522, num_heads=8, num_layers=6)

    def forward(self, x, tgt, mask, train):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)

        cnn_out = self.cnn(x)    # Expected shape: (batch_size, updated num channels, updated height, updated width)
        cnn_out = cnn_out.view(batch_size, seq_len, -1) 

        lstm_out, _ = self.lstm(cnn_out)   # Expected shape: (batch_size, sequence_length (num_frames), features)
        lstm_out = lstm_out.transpose(0, 1)   # Expected shape: (sequence_length (num_frames), batch_size, features)

        output = self.transformer(lstm_out, tgt, train=train)   # Expected shape: (target_sequence_length, batch_size, vocab_size)
        output = output.permute(1, 2, 0)
        
        return output


        # lstm_out = lstm_out.reshape(batch_size * 100 , -1)
        # # print("transposed LSTM OUT:", lstm_out.shape)

        # output = self.transformer(lstm_out.type(torch.int))   # Expected shape: (target_sequence_length, batch_size, vocab_size)
        # # print("transformer output:", output.shape)
        # # output = output.permute(1, 2, 0)     # Expected shape: (batch_size, vocab_size, target_sequence_length)  <-- needed for cross entropy loss
        # # output = torch.argmax(output["logits"], dim=-1)
        # output = output["logits"].view(batch_size, 100, -1)
        # output = output.permute(0, 2, 1)