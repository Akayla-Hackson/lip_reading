import torchvision.models as models
from torch import nn
from transformers import AutoTokenizer
import torch
import math
# class Transformer(nn.Module):
#     def __init__(self, feature_size=256, num_tokens=10000, num_heads=8, num_layers=6):
#         super().__init__()
#         self.embedding = nn.Embedding(num_tokens, feature_size)
#         transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads)
#         self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)
#         self.output_layer = nn.Linear(feature_size, num_tokens)

#     def forward(self, x, tgt):
#         tgt = self.embedding(tgt)     # Expected shape: (target_sequence_length, batch_size)
#         tgt = tgt.transpose(0, 1) # Expected shape: (target_sequence_length, batch_size, feature_size)
#         # print(f"Memory input shape: {x.shape}")    # Expected shape: (sequence_length, batch_size, feature_size)
#         out = self.transformer_decoder(tgt, x)
#         return self.output_layer(out)



class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int,  max_length: int = 5000):
      # d_model:      dimension of embeddings
      # dropout:      randomly zeroes-out some of the input
      # max_length:   max sequence length
    super().__init__()     

    pe = torch.zeros(max_length, d_model)    
    k = torch.arange(0, max_length).unsqueeze(1)  
    div_term = torch.exp(                                 
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(k * div_term)      
    pe[:, 1::2] = torch.cos(k * div_term)    
    pe = pe.unsqueeze(0)                              
    self.register_buffer("pe", pe)                        

  def forward(self, x):
    # x: embeddings (batch_size, seq_length, d_model)
    # Returns: embeddings + positional encodings (batch_size, seq_length, d_model)
    x = x + self.pe[:, : x.size(1)].requires_grad_(False) 
    return x


class Transformer(nn.Module):
    def __init__(self, feature_size=256, num_tokens=10000, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, feature_size)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(feature_size, num_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.pos_encoder = PositionalEncoding(feature_size)

    def forward(self, x, tgt, max_len=100, train=False):
        if not train:
            start_token_id = self.tokenizer.cls_token_id
            stop_token_id = self.tokenizer.sep_token_id
            vocab_size = self.tokenizer.vocab_size
            device = x.device
            outputs = torch.LongTensor([[start_token_id]]).to(device)
            
            # Initialize storage for logits with a fixed size, [max_len, 1, vocab_size]
            logits_history = torch.zeros(max_len, 1, vocab_size, device=device)

            for i in range(max_len):
                tgt_emb = self.embedding(outputs)
                tgt_emb = self.pos_encoder(tgt_emb)
                tgt_emb = tgt_emb.transpose(0, 1)
                out = self.transformer_decoder(tgt_emb, x)
                logits = self.output_layer(out)
                
                k=15
                top_logits, top_indices = torch.topk(logits[-1, :, :], k)
                top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices[0].tolist()] 
                current_output_tokens = self.tokenizer.decode(outputs[0].tolist())
                print(f"Current output tokens: {current_output_tokens}")
                print(f"Top {k} logits at step {i}: {top_logits}")
                print(f"Top {k} tokens: {top_tokens}")
                # exit()

                logits_history[i] = logits[-1, :, :]

                ####### Note: Maybe we should implement beam search or something???? ############################################
                # Predict next token based on logits
                next_token = torch.argmax(logits[-1, :, :], dim=-1, keepdim=True)

                # Stop if stop token is predicted
                if next_token.item() == stop_token_id:
                    print(f"Stop token ({stop_token_id}) predicted at step {i}")
                    break
                
                outputs = torch.cat((outputs, next_token.transpose(0, 1)), dim=1)
            
            # Trim unused pre-allocated space 
            logits_history = logits_history[:i+1]
            return logits_history
        else:
            tgt_emb = self.embedding(tgt)
            tgt_emb = self.pos_encoder(tgt_emb)
            tgt_emb = tgt_emb.transpose(0, 1)
            out = self.transformer_decoder(tgt_emb, x)
            return self.output_layer(out)
