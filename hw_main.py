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

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print(f"tokenitzer vocab size: {tokenizer.vocab_size}")
RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Pad each sequence to be the same length within the batch
def collate_fn(batch):
    sequences, labels = zip(*batch)
    encoded_labels = tokenizer(labels, add_special_tokens=True, max_length=100, padding="longest",  return_tensors='pt')
    
    return torch.unsqueeze(torch.stack(sequences[0]), axis=0), encoded_labels
# def collate_fn(batch):
#     sequences, labels = zip(*batch)
#     sequences = torch.stack(sequences) 
#     encoded_labels = tokenizer(labels, add_special_tokens=True, max_length=100, padding="longest", return_tensors='pt')
#     return sequences, encoded_labels


def main(args):
    now = datetime.datetime.now()
    # model = LipReadingModel()
    model = HwLipReadingModel()
    model.to(device)

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
        # scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=int(len(train_dataset)/args.grad_accum_steps))
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Outputs: [batch_size, vocab size, sequence_length] Targets: [batch_size, sequence_length]

        loss_history = []
        for epoch in range(args.epochs):
            model.train() 
            total_loss = 0 
            avg_loss = loss_accum = 0
            
            progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
            for batch_idx, (frames, targets) in progress_bar:
                frames, input_id, masks = frames.to(device, non_blocking=True), targets['input_ids'].to(device, non_blocking=True), targets['attention_mask'].bool().to(device, non_blocking=True)

                captions_in = input_id[:, :-1].to(device)
                captions_out = input_id[:, 1:].to(device)
                mask = (captions_out != tokenizer.pad_token_id).to(device)

                # print("captions in", captions_in)
                # print(tokenizer.batch_decode(captions_in, skip_special_tokens=False))
                # print("captions out", captions_out)
                # print(tokenizer.batch_decode(captions_out, skip_special_tokens=False))

                logits = model(frames, captions_in, args.train)
                logits = logits.permute(0, 2, 1)
                # loss = transformer_temporal_softmax_loss(logits, captions_out, mask)
                loss = criterion(logits, captions_out)

                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # total_loss += loss.item()
                loss_accum += loss
                loss = loss.detach()
                total_loss += loss
                avg_loss += loss

                if batch_idx != 0 and batch_idx % args.grad_accum_steps == 0:
                    optimizer.zero_grad()
                    loss_accum.backward()
                    plot_grad_flow(model.named_parameters())
                    del loss_accum
                    nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                    loss_accum = 0
                    optimizer.step()
                    scheduler.step()

                # writer.add_scalar('Training Loss', loss.item(), epoch * len(data_loader) + batch_idx)
                writer.add_scalar('Training Loss', loss, epoch * len(data_loader) + batch_idx)
                writer.add_scalar('Average Batch Loss', avg_loss/(batch_idx+1), epoch * len(data_loader) + batch_idx)

                print("target \n",tokenizer.batch_decode(input_id))
                print("Guess \n",tokenizer.batch_decode(torch.argmax(logits, dim=1)))
            average_loss = total_loss / len(data_loader)
            writer.add_scalar('Average Training Loss', average_loss, epoch)
            print(f"Average Loss for Epoch {epoch}: {average_loss}")
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
            torch.save(state, f"{epoch}.state")
    else:
        save_path = f'Val/{args.data_type}/Batch_size_{args.batch_size}/LR_{args.learning_rate}/Date_{now.month}_{now.day}_hr_{now.hour}'
        os.makedirs(save_path, exist_ok=True)
        writer = SummaryWriter(f'runs/{save_path}')

        model.load_state_dict(torch.load("0.state")['state_dict'])
        model.eval()
        print(model)

        val_dataset = LipReadingDataset(directory='./LRS2/data_splits/val' if os.getlogin() != "darke" else "D:/classes/cs231n/project/LRS2/data_splits/val", transform=None)
        val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        )

        total_words = 0
        total_wer = 0.0
        num_samples = 0
        print("Total samples loaded for validation:", len(val_data_loader))  

        for batch_idx, (frames, targets) in enumerate(val_data_loader):
            with torch.no_grad():
                frames, input_id, mask = frames.to(device, non_blocking=True), targets['input_ids'].to(device, non_blocking=True), targets['attention_mask'].to(device, non_blocking=True)
                print("target \n",tokenizer.batch_decode(input_id))
                output = model(frames, input_id, args.train)
                output = output.argmax(axis=1)
                
                predicted_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
                reference_texts = tokenizer.batch_decode(input_id, skip_special_tokens=True)
                
                # Calc WER for each item in the batch and accumulate
                for predicted, reference in zip(predicted_texts, reference_texts):
                    sample_wer = wer(reference, predicted)
                    total_wer += sample_wer
                    num_samples += 1
                # print("Predictions:", predicted_texts)
                # print("References:", reference_texts)

            print("target \n",tokenizer.batch_decode(input_id))
            print("Guess \n",tokenizer.batch_decode(output))
            print("WER:", sample_wer)
            writer.add_scalar('Val WER', sample_wer, len(val_data_loader) + batch_idx)
        
        # Calc avg WER across all samples
        average_wer = total_wer / num_samples
        print(f"Average WER: {average_wer:.2f}")
        writer.add_scalar('Average WER', average_wer)
        writer.add_scalar('Average WER', average_wer)

def one_hot_encoding(labels, num_classes):
    target = torch.zeros(labels.size(0), labels.size(1), num_classes, device=labels.device)
    # Scatter 1s into tensor according to labels
    target.scatter_(2, labels.unsqueeze(2), 1)
    return target

def transformer_temporal_softmax_loss(x, y, mask):

        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(x_flat,  y_flat, reduction='none')
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)

        return loss

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    ave_grads_cpu = [grad.cpu() for grad in ave_grads]
    plt.plot(ave_grads_cpu, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads_cpu)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads_cpu), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads_cpu))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()
# def plot_grad_flow(named_parameters):
#     '''Plots the gradients flowing through different layers in the net during training.
#     Can be used for checking for possible gradient vanishing / exploding problems.
    
#     Usage: Plug this function in Trainer class after loss.backwards() as 
#     "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
#     ave_grads = []
#     max_grads= []
#     layers = []
#     for n, p in named_parameters:
#         if(p.requires_grad) and ("bias" not in n):
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean())
#             max_grads.append(p.grad.abs().max())
#     max_grads_cpu = [grad.cpu() for grad in max_grads]
#     ave_grads_cpu = [grad.cpu() for grad in ave_grads]
#     plt.bar(np.arange(len(max_grads_cpu)), max_grads_cpu, alpha=0.1, lw=1, color="c")
#     plt.bar(np.arange(len(ave_grads_cpu)), ave_grads_cpu, alpha=0.1, lw=1, color="b")
#     plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
#     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
#     plt.xlim(left=0, right=len(ave_grads))
#     plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
#     plt.xlabel("Layers")
#     plt.ylabel("average gradient")
#     plt.title("Gradient flow")
#     plt.grid(True)
#     plt.legend([Line2D([0], [0], color="c", lw=4),
#                 Line2D([0], [0], color="b", lw=4),
#                 Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
#     plt.show() 
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