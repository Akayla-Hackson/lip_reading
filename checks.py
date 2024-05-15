import torch
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from classes.dataset import LipReadingDataset
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torch import nn
import numpy as np

def check_data_normalization(data_loader):
    for frames, _ in data_loader:
        print("Max pixel value:", frames.max().item())
        print("Min pixel value:", frames.min().item())
        print("Mean pixel value:", frames.float().mean().item())
        break 



def visualize_frames(data_loader, tokenizer):
    frames, labels = next(iter(data_loader))
    frames = frames.squeeze(0)

    # Decode the label for the sequence
    decoded_label = tokenizer.decode(labels[0], skip_special_tokens=True)

    fig, axes = plt.subplots(nrows=1, ncols=min(4, frames.shape[0]), figsize=(15, 5)) 
    for i, ax in enumerate(axes.flat):
        frame_to_show = frames[i].permute(1, 2, 0) 
        ax.imshow(frame_to_show.cpu().numpy())  
        ax.set_title(f"Frame {i+1}")
        ax.axis('off')
    
    fig.suptitle(f"Label for sequence: {decoded_label}", fontsize=16)
    plt.subplots_adjust(top=0.85)  
    plt.show()  


def collate_fn(batch):
    sequences, labels = zip(*batch)
    # Pad frames
    padded_sequences = pad_sequence([torch.stack(seq) for seq in sequences], batch_first=True, padding_value=0.0)
    # Tokenize and pad labels
    encoded_labels = [tokenizer.encode(label, add_special_tokens=True, max_length=512, truncation=True) for label in labels]
    padded_labels = pad_sequence([torch.tensor(label, dtype=torch.long) for label in encoded_labels], batch_first=True, padding_value=tokenizer.pad_token_id)
    
    return padded_sequences, padded_labels




train_dataset = LipReadingDataset(directory='./LRS2/data_splits/train' if os.getlogin() != "darke" else "D:/classes/cs231n/project/LRS2/data_splits/train", transform=None)
print("Total samples loaded:", len(train_dataset))  
data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    # shuffle=True,
    collate_fn=collate_fn,
    )
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"tokenitzer vocab size: {tokenizer.vocab_size}")


check_data_normalization(data_loader)
visualize_frames(data_loader, tokenizer)