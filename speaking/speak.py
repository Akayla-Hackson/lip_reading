import torch
from torch import nn
from torch.nn import Conv3d
import numpy as np

class Conv3d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, bias=True, residual=False, 
                    *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv3d(cin, cout, kernel_size, stride, padding, bias=bias),
                            nn.BatchNorm3d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

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

class Speaking_conv3d_layers(nn.Module):
    def __init__(self, d_model, linear_size):
        super().__init__()
        self.linear_size = linear_size 
        self.d_model = d_model

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
        
        self.linear = nn.Linear(d_model*3*3*linear_size, 2*linear_size)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view((x.shape[0], -1))
        # size = self.d_model*3*3*self.linear_size
        # padded_remainder_to_x_match_shape = torch.zeros(size - x.shape[1]).unsqueeze(axis=0).to(x.device)
        # padded_remainder_to_y_match_shape = torch.zeros(self.linear_size - y.shape[1]).unsqueeze(axis=0).to(x.device)

        # x  = torch.cat((x, padded_remainder_to_x_match_shape), axis=1)
        # y  = torch.cat((y, padded_remainder_to_y_match_shape), axis=1)
        
        x = self.linear(x)
        x = x.view(x.shape[0], 2, -1)
        return x


# def CNN_Baseline(vocab, visual_dim, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, 
#                     backbone=True):
#     c = copy.deepcopy

#     attn = MultiHeadedAttention(h, d_model, dropout=dropout)
#     ff = PositionwiseFeedForward(d_model, d_ff, dropout)
#     position = PositionalEncoding(d_model, dropout)

#     model = EncoderDecoder(
#         (CNN_3d(visual_dim) if backbone else None),
#         Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
#         Decoder(DecoderLayer(d_model, c(attn), c(attn), 
#                              c(ff), dropout), N),

#         nn.Sequential(c(position)),
#         nn.Sequential(Embeddings(d_model, vocab), c(position)),
#         Generator(d_model, vocab))

#     # Initialize parameters with Glorot / fan_avg.
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     return model