import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from classes.models import *
from classes.modules import *
from classes.paper_lipreading import *
from classes.search import *

pad_token = '<pad>'
# start_symbol, end_symbol = 100, 101
start_symbol, end_symbol = 101, 102
class LipReadingModel(nn.Module):
    def __init__(self, vocab_size, feat_dim, hidden_units, num_heads, num_blocks, dropout_rate, device, lm_alpha=0.0):
        super(LipReadingModel, self).__init__()
        self.vocab_size = vocab_size
        self.feat_dim = feat_dim
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.device = device
        self.lm_alpha = lm_alpha

        # Initialize the core model components
        self.model = VTP_24x24(
            vocab=vocab_size + 1, 
            visual_dim=feat_dim, 
            N=num_blocks, 
            d_model=hidden_units, 
            d_ff=2048,  
            h=num_heads, 
            dropout=dropout_rate, 
            backbone=True 
        ).to(device)
        if lm_alpha > 0:
            self.lm = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_fast=True)
            self.lm_tokenizer.pad_token = '<pad>'
        else:
            self.lm = None
            self.lm_tokenizer = None

    @autocast()
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, train=True):
        if train:
            encoder_output, src_mask = self.model.encode(src, src_mask)
            if tgt is None:
                raise ValueError("tgt must be provided for training.")

            logits = self.model.decode(encoder_output, tgt, src_mask, tgt_mask)
            return logits
        else: 
            beam_outs, beam_scores = beam_search(
                decoder=self.model,
                bos_index=start_symbol,
                eos_index=end_symbol,
                max_output_length=200, 
                pad_index=0,
                encoder_output=encoder_output,
                src_mask=src_mask,
                size=5,  
                alpha=1.0,  
                n_best=5  # Return the top-5 beams
            )
            return beam_outs, beam_scores


