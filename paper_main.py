import torch
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from classes.dataset import LipReadingDataset
from classes.cnn import CNN
from classes.lstm import LSTM
from classes.transformer import Transformer
from classes.lip_reading import LipReadingModel
from classes.hw_lip_reading import HwLipReadingModel
from classes.hw_transformer import *
from classes.hw_transformer_layers import *
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
import datetime
import numpy as np
import random
from torch import nn
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from jiwer import wer
from matplotlib.lines import Line2D
from transformers import GPT2Tokenizer
from classes.models import *
from classes.modules import *
from classes.paper_lipreading import *
from classes.search import *

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# print(f"tokenitzer vocab size: {tokenizer.vocab_size}")

# pad_token = '<pad>'
# bos_token = '<bos>'
# eos_token = '<eos>'
# unk_token = '<unk>'

pad_token = 0
bos_token = 101
eos_token = 102


# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = pad_token

# tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', cache_dir='checkpoints/tokenizers', use_fast=True)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='checkpoints/tokenizers', use_fast=True)

# start_symbol, end_symbol = 100, 101

RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Pad each sequence to be the same length within the batch
def collate_fn(batch):
    sequences, labels = zip(*batch)
    encoded_labels = tokenizer(labels, add_special_tokens=True, max_length=100, padding="longest",  return_tensors='pt')
    return torch.unsqueeze(torch.stack(sequences[0]), axis=0), encoded_labels



def main(args):
    now = datetime.datetime.now()
    model = LipReadingModel(
        vocab_size=tokenizer.vocab_size,  
        feat_dim=512,  # Feature dim
        hidden_units=512,  # Size of the hidden layers
        num_heads=8,  # Number of attention heads
        num_blocks=6,  # Number of transformer blocks
        dropout_rate=0.1,  
        device=device,
        lm_alpha=0.1  # Influence of the language model 
    )
    model.to(device)
    model.train()

    if args.train:
        save_path = f'{args.data_type}/Batch_size_{args.batch_size}/LR_{args.learning_rate}/Date_{now.month}_{now.day}_hr_{now.hour}'
        os.makedirs(save_path, exist_ok=True)
        writer = SummaryWriter(f'runs/{save_path}')

        train_dataset = LipReadingDataset(directory='./LRS2/data_splits/train' if os.getlogin() != "darke" else "D:/classes/cs231n/project/LRS2/data_splits/train", transform=None)
        print("Total samples loaded:", len(train_dataset))  
        data_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=False,
            )
        
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.99, 0.95), lr=args.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=int(len(train_dataset)/args.grad_accum_steps))
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Outputs: [batch_size, vocab size, sequence_length] Targets: [batch_size, sequence_length]

        loss_history = []
        for epoch in range(args.epochs):
            model.train() 
            total_loss = 0 
            avg_loss = loss_accum = 0
            
            progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
            for batch_idx, (frames, targets) in progress_bar:
                frames = frames.to(device)
                input_id = targets['input_ids'].to(device)
                attention_mask = targets['attention_mask'].to(device)

                frames_mask = torch.ones(frames.shape[0], frames.shape[1], dtype=torch.long).to(device)  # mask is all 1's since no padding is added (batch_size = 1)

                max_length = frames.shape[1]  # frames is [batch, frames, channels, height, width]
                # Pad input_ids and attention_mask to match the frames length
                if input_id.shape[1] < max_length:
                    padding_needed = max_length - input_id.shape[1]

                    padded_input_id = F.pad(input_id, pad=(0, padding_needed), value=tokenizer.pad_token_id)
                    padded_attention_mask = F.pad(attention_mask, pad=(0, padding_needed), value=0)
                    input_id = padded_input_id.to(device)
                    attention_mask = padded_attention_mask.to(device)

                outputs = model(frames, input_id, frames_mask, attention_mask, args.train)
                outputs = outputs.permute(0,2,1)
                loss = criterion(outputs, input_id)

                loss_accum += loss
                loss = loss.detach()
                total_loss += loss
                avg_loss += loss

                if batch_idx != 0 and batch_idx % args.grad_accum_steps == 0:
                    optimizer.zero_grad()
                    loss_accum.backward()
                    del loss_accum
                    nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                    loss_accum = 0
                    optimizer.step()
                    scheduler.step()

                writer.add_scalar('Training Loss', loss, epoch * len(data_loader) + batch_idx)
                writer.add_scalar('Average Batch Loss', avg_loss/(batch_idx+1), epoch * len(data_loader) + batch_idx)

                print("target \n",tokenizer.batch_decode(input_id))
                print("Guess \n",tokenizer.batch_decode(torch.argmax(outputs, dim=1)))
            average_loss = total_loss / len(data_loader)
            writer.add_scalar('Average Training Loss', average_loss, epoch)
            print(f"Average Loss for Epoch {epoch}: {average_loss}")
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
            torch.save(state, f"{epoch}.state")


def levenshtein(a, b):
  """Calculates the Levenshtein distance between a and b.
  The code was taken from: http://hetland.org/coding/python/levenshtein.py
  """
  n, m = len(a), len(b)
  if n > m:
    # Make sure n <= m, to use O(min(n,m)) space
    a, b = b, a
    n, m = m, n
  current = list(range(n + 1))
  for i in range(1, m + 1):
    previous, current = current, [i] + [0] * n
    for j in range(1, n + 1):
      add, delete = previous[j] + 1, current[j - 1] + 1
      change = previous[j - 1]
      if a[j - 1] != b[i - 1]:
        change = change + 1
      current[j] = min(add, delete, change)
  return current[n]


if __name__ == "__main__":
    # device = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type', default='train', type=str, help='dataset used for training')
    parser.add_argument('--batch_size', default=1, type=int, help='num entries per batch')
    parser.add_argument('--grad_accum_steps', default=16, type=int, help='How many steps to acumulate grad')
    parser.add_argument('--train', default=True, type=bool, help='Train or eval')


    parser.add_argument('--num_workers', default=4, type=int, help='num of workes for the dataloader')

    parser.add_argument('--learning_rate', default=0.001, type=int, help='learning rate for optimizer')
    # 3e-4 
    parser.add_argument('--epochs', default=1, type=int, help='num epoch to train for')
    args = parser.parse_args()
    main(args)