from  torch.nn import functional as F
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

class Conv3d_trans(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, bias=True, residual=False, 
                    *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose3d(cin, cout, kernel_size, stride, padding, bias=bias),
                            nn.BatchNorm3d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class Conv3d_res(nn.Module):
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
        return self.act(out), x

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

class ConvEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
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
            Conv3d(512, d_model, kernel_size=(1, 3, 3), stride=1, padding=(0, 0, 0)),
            )
    
    
    def forward(self, x):
        x = self.encoder(x)
        return x

    
class Speaking_conv3d_layers(nn.Module):
    def __init__(self, d_model, length_of_video_in_frames):
        super().__init__()
        self.linear_size = length_of_video_in_frames 
        self.d_model = d_model
        self.encoder = ConvEncoder(d_model)
        
        self.linear = nn.Linear(d_model*3*3*length_of_video_in_frames, 2*length_of_video_in_frames) # output size is 1 or zero times the temporal dimension

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

class Speaking_words(nn.Module):
    def __init__(self, d_model, length_of_video_in_frames, tokenizer_vocab):
        super().__init__()
        self.linear_size = length_of_video_in_frames 
        self.d_model = d_model
        self.tokenizer_vocab = tokenizer_vocab
        self.conv1 = Conv3d_res(3, 64, kernel_size=5, stride=(1, 2, 2), padding=2)  # 80, 80
        self.conv2 = Conv3d_res(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)) # 40, 40
        self.conv3 = Conv3d_res(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True)
        self.conv4 = Conv3d_res(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)) # 20, 20
        self.conv5 = Conv3d_res(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True) 
        self.conv6 = Conv3d_res(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)) # 10, 10
        self.conv7 = Conv3d_res(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True)
        self.conv8 = Conv3d_res(512, d_model, kernel_size=(1, 3, 3), stride=1, padding=(0, 0, 0)) # 8, 8
        self.conv9 = Conv3d_res(d_model, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 0, 0)) # 6, 6

    
        
        self.linear = nn.Linear(128*6*6*length_of_video_in_frames, 128)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(128, tokenizer_vocab*length_of_video_in_frames) # output size is 1 or zero times the temporal dimension

    def forward(self, x):
        x, res1 = self.conv1(x)
        x, res2 = self.conv2(x)
        x, res3 = self.conv3(x) 
        x, res4 = self.conv4(x)
        x, res5 = self.conv5(x)
        x, res6 = self.conv6(x)
        x, res7 = self.conv7(x)
        x, res8 = self.conv8(x)
        x, res8 = self.conv9(x)


        x = x.view((x.shape[0], -1))
        # size = self.d_model*3*3*self.linear_size
        # padded_remainder_to_x_match_shape = torch.zeros(size - x.shape[1]).unsqueeze(axis=0).to(x.device)
        # padded_remainder_to_y_match_shape = torch.zeros(self.linear_size - y.shape[1]).unsqueeze(axis=0).to(x.device)

        # x  = torch.cat((x, padded_remainder_to_x_match_shape), axis=1)
        # y  = torch.cat((y, padded_remainder_to_y_match_shape), axis=1)
        
        x = self.gelu(self.linear(x))
        x = self.linear2(x)
        x = x.view(x.shape[0], self.tokenizer_vocab, -1)
        
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