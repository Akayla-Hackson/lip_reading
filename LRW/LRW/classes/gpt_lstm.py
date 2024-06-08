import torchvision.models as models
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # For bidirectional LSTM
        self.fc = nn.Linear(256, 768)

    def forward(self, x, target_length):
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = self.fc(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)  # permute to [batch_size, hidden_size, seq_len] for pooling
        adaptive_pooling = nn.AdaptiveMaxPool1d(target_length)  # create adaptive pooling layer
        pooled_out = adaptive_pooling(lstm_out)  # pooled_out: [batch_size, hidden_size, target_length]
        
        pooled_out = pooled_out.permute(0, 2, 1) 
        return pooled_out