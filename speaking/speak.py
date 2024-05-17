import torch
from torch import nn
from torch.nn import Conv3d



class Speaking_model(nn.Module):
    def __init__(self):
        self.conv = nn.Conv3d(3, )
        self.linear = nn.Linear(100, 2)
        self.norm = nn.BatchNorm3d(100)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.linear(x)
        return x
    
 self.encoder = nn.Sequential(
            Conv3d(3, 64, kernel_size=5, stride=(1, 2, 2), padding=2),  # 48, 48

            Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 24, 24
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 12, 12
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 6, 6
            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 3, 3
            Conv3d(512, d_model, kernel_size=(1, 3, 3), stride=1, padding=(0, 0, 0)),)

def CNN_Baseline(vocab, visual_dim, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, 
                    backbone=True):
    c = copy.deepcopy

    attn = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        (CNN_3d(visual_dim) if backbone else None),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),

        nn.Sequential(c(position)),
        nn.Sequential(Embeddings(d_model, vocab), c(position)),
        Generator(d_model, vocab))

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model