import numpy as np
import copy

import torch
import torch.nn as nn

from classes.hw_transformer_layers import *
from transformers import AutoTokenizer

class LipReadingTransformer(nn.Module):
    def __init__(self, feature_size, num_heads=4, num_layers=2, max_length=200):
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.vocab_size = tokenizer.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, feature_size, padding_idx=tokenizer.pad_token_id)
        self.positional_encoding = PositionalEncoding(feature_size, max_len=max_length)
        
        decoder_layer = TransformerDecoderLayer(input_dim=feature_size, num_heads=num_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)

        self.output = nn.Linear(feature_size, self.vocab_size)

    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):

        # print("features shape:", features.shape)
        # print("captions shape:", captions.shape)
        batch_size, seq_len = captions.shape # Expected shape: (batch_size, sequence_length)

        scores = torch.empty((batch_size, seq_len, self.vocab_size))

        captions_embedded = self.embedding(captions)  
        captions_embedded = self.positional_encoding(captions_embedded) 
        # print("captions shape after embedding and positional encoding", captions.shape)

        tgt_mask = torch.tril(torch.ones((seq_len, seq_len), device=captions.device)).bool()
        # print("captions mask shape", tgt_mask.shape) 
        # print("captions mask", tgt_mask) 

        transformer_output = self.transformer(captions_embedded, features, tgt_mask=tgt_mask)
        # print("transformer output shape", transformer_output.shape) 

        scores = self.output(transformer_output) 
        # print("transformer scores shape", scores.shape) 
        # exit()

        return scores

    
    # STILL MUST TEST
    def sample(self, features, max_length=30):
        N, _, _ = features.shape
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        partial_caption = self._start * np.ones(N, dtype=np.int32)
        partial_caption = torch.LongTensor(partial_caption).unsqueeze(1)

        for t in range(max_length):
            output_logits = self.forward(features, partial_caption)
            output_logits = output_logits[:, -1, :]
            word = torch.argmax(output_logits, axis=1)
            captions[:, t] = word.numpy()
            word = word.unsqueeze(1)
            partial_caption = torch.cat([partial_caption, word], dim=1)

        return captions

class TransformerDecoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder, to be used with TransformerDecoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerDecoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.multihead_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


    def forward(self, tgt, memory, tgt_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer.

        Inputs:
        - tgt: the sequence to the decoder layer, of shape (N, T, W)
        - memory: the sequence from the last layer of the encoder, of shape (N, S, D)
        - tgt_mask: the parts of the target sequence to mask, of shape (T, T)

        Returns:
        - out: the Transformer features, of shape (N, T, W)
        """
        # Perform self-attention on the target sequence (along with dropout and
        # layer norm).
        tgt2 = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Attend to both the target sequence and the sequence from the last
        # encoder layer.
        tgt2 = self.multihead_attn(query=tgt, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Pass
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)

        return output
