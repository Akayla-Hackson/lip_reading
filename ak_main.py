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
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import random
from torch import nn

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Pad each sequence to be the same length within the batch
def collate_fn(batch):
    sequences, labels = zip(*batch)
    # Pad frames
    padded_sequences = pad_sequence([torch.stack(seq) for seq in sequences], batch_first=True, padding_value=0.0)
    # Tokenize and pad labels
    encoded_labels = [tokenizer.encode(label, add_special_tokens=True, max_length=512, truncation=True) for label in labels]
    padded_labels = pad_sequence([torch.tensor(label) for label in encoded_labels], batch_first=True, padding_value=tokenizer.pad_token_id)
    
    return padded_sequences, padded_labels

def encode_labels(labels):
    return [tokenizer.encode(label, add_special_tokens=True) for label in labels]


def main(args):
    now = datetime.datetime.now()
    save_path = f'{args.data_type}/Batch_size_{args.batch_size}/LR_{args.learning_rate}/Date_{now.month}_{now.day}_hr_{now.hour}'
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(f'runs/{save_path}')

    train_dataset = LipReadingDataset(directory='./LRS2/data_splits/train' if os.getlogin() != "darke" else "D:/classes/cs231n/project/LRS2/data_splits/train", transform=None)
    print("Total samples loaded:", len(train_dataset))  # Debug: Output the total number of samples loaded
    data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
        )

    model = LipReadingModel()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train() 
        total_loss = 0
            
        for batch_idx, (frames, targets) in enumerate(data_loader):
            frames, targets = frames.to(device), targets.to(device)
            # Reset gradients
            optimizer.zero_grad()
            
            output = model(frames, targets)
            targets = targets.type(torch.float32)
            loss = criterion(output, targets.view(-1)) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            writer.add_scalar('Training Loss', loss.item(), epoch * len(data_loader) + batch_idx)

        # Avg loss for the epoch
        average_loss = total_loss / len(data_loader)
        writer.add_scalar('Average Training Loss', average_loss, epoch)
        print(f"Average Loss for Epoch {epoch}: {average_loss}")


    # val_dataset = LipReadingDataset(directory='./LRS2/data_splits/val', transform=None)
    # validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)




if __name__ == "__main__":
    device = 'cpu'
    print(device)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type', default='train', type=str, help='dataset used for training')
    parser.add_argument('--batch_size', default=1, type=int, help='num entries per batch')

    parser.add_argument('--learning_rate', default=0.001, type=int, help='learning rate for optimizer')
    parser.add_argument('--epochs', default=10, type=int, help='num epoch to train for')
    args = parser.parse_args()
    main(args)
