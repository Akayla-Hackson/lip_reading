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

    def forward(self, x, tgt, mask):
        # print("\nX shape:", x.shape)
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        # print("X New shape:", x.shape)


        cnn_out = self.cnn(x)    # Expected shape: (batch_size, updated num channels, updated height, updated width)
        # print("\nCNN OUT:", cnn_out.shape)
        cnn_out = cnn_out.view(batch_size, seq_len, -1) 
        # print("New CNN OUT:", cnn_out.shape)

        lstm_out, _ = self.lstm(cnn_out)   # Expected shape: (batch_size, sequence_length (num_frames), features)
        # print("\nLSTM OUT:", lstm_out.shape)
        # Transpose LSTM output to match Transformer input expectation
        lstm_out = lstm_out.transpose(0, 1)   # Expected shape: (sequence_length (num_frames), batch_size, features)
        # print("transposed LSTM OUT:", lstm_out.shape)

        output = self.transformer(lstm_out, tgt)   # Expected shape: (target_sequence_length, batch_size, vocab_size)
        # print("transformer output:", output.shape)
        output = output.permute(1, 2, 0)
        # print("final output:", output.shape)


        return output


        # lstm_out = lstm_out.reshape(batch_size * 100 , -1)
        # # print("transposed LSTM OUT:", lstm_out.shape)

        # output = self.transformer(lstm_out.type(torch.int))   # Expected shape: (target_sequence_length, batch_size, vocab_size)
        # # print("transformer output:", output.shape)
        # # output = output.permute(1, 2, 0)     # Expected shape: (batch_size, vocab_size, target_sequence_length)  <-- needed for cross entropy loss
        # # output = torch.argmax(output["logits"], dim=-1)
        # output = output["logits"].view(batch_size, 100, -1)
        # output = output.permute(0, 2, 1)