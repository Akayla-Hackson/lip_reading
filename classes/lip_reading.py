import torchvision.models as models
from torch import nn

class LipReadingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = LSTMModel(input_dim=512, hidden_dim=256, num_layers=1)  
        self.transformer_decoder = TransformerDecoderModel(feature_size=256, num_tokens=1000, num_heads=8, num_layers=6)

    def forward(self, x, tgt):
        cnn_out = self.cnn(x)
        lstm_out = self.lstm(cnn_out)
        output = self.transformer_decoder(lstm_out, tgt)
        return output
