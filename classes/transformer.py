import torchvision.models as models
from torch import nn

class Transformer(nn.Module):
    def __init__(self, feature_size, num_tokens, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, feature_size)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(feature_size, num_tokens)

    def forward(self, x, tgt):
        tgt = self.embedding(tgt)
        out = self.transformer_decoder(tgt, x)
        return self.output_layer(out)
