import torchvision.models as models
from torch import nn

class Transformer(nn.Module):
    def __init__(self, feature_size=256, num_tokens=10000, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, feature_size)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(feature_size, num_tokens)

    def forward(self, x, tgt):
        tgt = self.embedding(tgt)     # Expected shape: (target_sequence_length, batch_size)
        tgt = tgt.transpose(0, 1) # Expected shape: (target_sequence_length, batch_size, feature_size)
        # print(f"NEW tgt embedding shape: {tgt.shape}")
        # print(f"Memory input shape: {x.shape}")    # Expected shape: (sequence_length, batch_size, feature_size)
        out = self.transformer_decoder(tgt, x)
        return self.output_layer(out)
