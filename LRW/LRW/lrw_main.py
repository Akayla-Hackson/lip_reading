import torch
from classes.ctc_decoder import Decoder
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
from classes.LRWDataset import LRWDataset
from optim_utils import CosineScheduler
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F


def collate_fn(batch):
    xs, ys, lens, indices = zip(*batch)
    max_len = max(lens)
    x = default_collate(xs)
    
    # x.narrow(2, 0, max_len)
    y = []
    for sub in ys: y += sub
    y = torch.IntTensor(y)
    
    lengths = torch.IntTensor(lens)
    y_lengths = torch.IntTensor([len(label) for label in ys])
    ids = default_collate(indices)
    
    return x, y, lengths, y_lengths, ids

    # max_length = max(len(sample['video']) for sample in batch)
    # batch_size = len(batch)
    # c, h, w = batch[0]['video'][0].shape

    # videos = torch.zeros((batch_size, max_length, c, h, w))
    # labels = []
    # input_lengths = []
    # target_lengths = []

    # for i, sample in enumerate(batch):
    #     video = sample['video']
    #     length = len(video)
        
    #     videos[i, :length, :, :, :] = video
    #     labels.extend(sample['label'].tolist())
    #     input_lengths.append(torch.tensor(length))
    #     target_lengths.append(len(sample['label']))

    # labels = torch.IntTensor(labels)
    # return {'video': videos, 'label': labels, 'input_lengths': torch.stack(input_lengths), 'target_lengths': torch.tensor(target_lengths, dtype=torch.long)}
    # sequences, labels = zip(*batch)
    # print(batch)
    # return torch.stack(sequences[0]), labels
def save_model(model, optimizer, args, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_lrw(args):
    now = datetime.datetime.now()
    save_path = f'{args.data_type}/Batch_size_{args.batch_size}/LR_{args.lr}/Date_{now.month}_{now.day}_hr_{now.hour}'
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(f'runs/{save_path}')
    train_dataset = LRWDataset(directory='./data_splits/train')
    vocab = train_dataset.vocab
    print ('vocab = {}'.format('|'.join(train_dataset.vocab)))  
    print("Total samples loaded:", len(train_dataset))  

    model = LipReadingModel(vocab)
    model.to(device)
    criterion = nn.CTCLoss(reduction='none', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineScheduler(args.lr, args.epochs)
    decoder = Decoder(vocab)
    train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=False,
    )
    best_wer = 1.0
    predictions, gt = [], []
    def predict(logits, y, lengths, y_lengths, n_show=2, mode='greedy'):
        n = min(n_show, logits.size(1))
        if mode == 'greedy':
            decoded = decoder.decode_greedy(logits, lengths)
        elif mode == 'beam':
            decoded = decoder.decode_beam(logits, lengths)

        predictions.extend(decoded)

        cursor = 0
        for b in range(x.size(0)):
            y_str = ''.join([vocab[ch - 1] for ch in y[cursor: cursor + y_lengths[b]]])
            gt.append(y_str)
            cursor += y_lengths[b]
            if b < n:
                print ('Test seq {}: {}; pred_{}: {}'.format(b + 1, y_str, mode, decoded[b]))


    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        num_batches = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
        for i, batch in progress_bar:
            optimizer.zero_grad()
            x, y, lengths, y_lengths, idx = batch

            x, y, lengths, y_lengths = x.to(device), y.to(device), lengths.to(device), y_lengths.to(device)
            logits = model(x)
            logits = logits.transpose(0, 1)
            
            with torch.backends.cudnn.flags(enabled=False):
                loss_all = criterion(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
            loss = loss_all.mean()

            # with torch.backends.cudnn.flags(enabled=False):
            #     loss = criterion(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
            if torch.isnan(loss).any():
                print ('Skipping iteration with NaN loss')
                continue

            weight = torch.ones_like(loss_all)
            dlogits = torch.autograd.grad(loss_all, logits, grad_outputs=weight)[0]
            # can add gradient clipping 
            logits.backward(dlogits)
   
            optimizer.step()
            scheduler.adjust_lr(optimizer, epoch)
            total_loss += loss.item()
            num_batches += 1
            # [29, 16, 28]
            # print("logits shape", logits.shape)
            # print("input length:", lengths)
            predict(logits, y, lengths, y_lengths, mode='beam')
        
        print(f"Average Loss for Epoch {epoch}: {loss.item():.4f}")
        writer.add_scalar('Loss/train', loss.item(), epoch)
       
        # Calculate accuracy
        correct_predictions = sum([1 for ground_truth, pred in zip(gt, predictions) if ground_truth == pred])
        accuracy = correct_predictions / len(predictions) * 100
        print(f"Accuracy: {accuracy}%")
        wer_metric = decoder.wer_batch(predictions, gt)
        cer_metric = decoder.cer_batch(predictions, gt)
        print("WER:", wer_result)
        print("CER:", wer_result)
        # save_model(model, optimizer, args, args.filepath)
        # if wer_metric < best_wer:
        #     best_wer = wer_metric
        #     # save_model(model, optimizer, args, args.filepath)
        #     state = {
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         }
        #     torch.save(state, f"{epoch}.state")

    model.eval()
    with torch.no_grad():
        val_dataset = LRWDataset(directory='./data_splits/val', test=True)
        print("Total val samples loaded:", len(val_dataset))  
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        val_predictions, val_gt = [], []
        progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for i, batch in progress_bar:
            x, y, lengths, y_lengths, idx = batch

            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits = logits.transpose(0, 1)
            
            with torch.backends.cudnn.flags(enabled=False):
                loss_all = criterion(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
            loss = loss_all.mean()
            if torch.isnan(loss).any():
                print ('Skipping iteration with NaN loss')
                continue

            predict(logits, y, lengths, y_lengths, mode='beam')
        print(f"test loss: {loss.item():.4f}")
        wer_result = decoder.wer_batch(predictions, gt)
        print("WER:", wer_result)    

def test_lrw(args):
    model.eval()
    with torch.no_grad():
        
        test_dataset = LRWDataset(directory='./LRW/data_splits/val')
        vocab = test_dataset.vocab
    
        print("Total samples loaded:", len(test_dataset))  

        model = LipReadingModel(vocab)
        saved = torch.load(args.filepath)
        model.load_state_dict(saved['model'])
        model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        criterion = nn.CTCLoss(reduction='none', zero_infinity=True)

        decoder = Decoder(vocab)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        best_wer = 1.0
        predictions, gt = [], []
        
        progress_bar = tqdm(enumerate(test_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
        for i, batch in progress_bar:
            x, y, lengths, y_lengths, idx = batch

            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits = logits.transpose(0, 1)
            
            with torch.backends.cudnn.flags(enabled=False):
                loss_all = criterion(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
            loss = loss_all.mean()
            if torch.isnan(loss).any():
                print ('Skipping iteration with NaN loss')
                continue

            predict(logits, y, lengths, y_lengths, mode='beam')
        
        print(f"test loss: {loss.item():.4f}")
        wer_result = decoder.wer_batch(predictions, gt)
        print("WER:", wer_result)
        

if __name__ == "__main__":
   
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # elif torch.backends.mps.is_available():      
    #     device = 'mps'                         
    else:
        device = torch.device('cpu')
       
    print(device)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type', default='train', type=str, help='dataset used for training')
    parser.add_argument('--batch_size', default=16, type=int, help='num entries per batch')
    parser.add_argument('--grad_accum_steps', default=16, type=int, help='How many steps to acumulate grad')
    parser.add_argument('--train', default=True, type=bool, help='Train or eval')


    parser.add_argument('--num_workers', default=1, type=int, help='num of workes for the dataloader')

    parser.add_argument('--lr', default=3e-4, type=int, help='learning rate for optimizer')
    # 3e-4 
    parser.add_argument('--epochs', default=10, type=int, help='num epoch to train for')
    args = parser.parse_args()
    args.filepath = f'{args.epochs}-{args.lr}-lrw.pt' # Save path.
    if args.train:
        train_lrw(args)
    else:    
        test_lrw(args)
