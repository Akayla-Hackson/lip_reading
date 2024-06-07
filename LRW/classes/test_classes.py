import torch
from torchvision import transforms
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


def save_batch_images(images, batch_idx, n_max=5, save_dir='saved_images'):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axs = plt.subplots(1, n_max, figsize=(15, 5))
    for i, img in enumerate(images[:n_max]):
        axs[i].imshow(img.permute(1, 2, 0))  # Permute the tensor to (H,W,C)
        axs[i].axis('off')
    
    plt.savefig(f'{save_dir}/batch_{batch_idx+1}.png')
    plt.close()

def test_data_loader():

    # # Transformation For ResNet
    # transform = transforms.Compose([
    #     transforms.Resize(256),  # Resize to slightly larger square
    #     transforms.CenterCrop(224),  # Crop to 224x224
    #     transforms.ToTensor(),  # Convert the image to tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
    # ])

    transform = None
    dataset = LipReadingDataset(directory='./LRS2/data_splits/train', transform=transform)
    print("Total samples loaded:", len(dataset))  # Debug: Output the total number of samples loaded

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for batch_idx, (padded_frames, padded_labels) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}")
        print("Padded frames shape:", padded_frames.shape)
        print("Padded labels shape:", padded_labels.shape)
        print("Padded labels:", padded_labels)
        decoded_texts = [tokenizer.decode(ids) for ids in padded_labels]
        print("Decoded labels:", decoded_texts)
        if batch_idx == 0:  # Limit to checking a few batches
            break
        

        # Save first few images of each batch
        for i in range(len(padded_frames)):
            print(f"Label {i+1}: {padded_labels[i]}")
            save_batch_images(padded_frames[i], batch_idx, save_dir='saved_images')

        # Limit to inspecting a few batches
        if batch_idx == 0:
            break



def test_cnn():
    # CNN expects a 3-channel image
    cnn = CNN()
    dummy_input = torch.rand(1, 3, 128, 128)  # Ex input tensor (batch size, channels, height, width)
    output = cnn(dummy_input)
    print("CNN Output Shape:", output.shape) # Shape: (batch size, updated num channels, updated height, updated width)



def test_lstm():
    # LSTM expects input (batch, seq_len, features)
    lstm = LSTM(input_dim=512, hidden_dim=256, num_layers=1)
    dummy_input = torch.rand(1, 10, 512)  # Ex input tensor (batch size, sequence length, features)
    output, (hidden, cell) = lstm(dummy_input)
    print("LSTM Output Shape:", output.shape)   # Shape: (batch size, sequence length, features)
    print("LSTM Hidden State Shape:", hidden.shape)   # Shape: (num lstm layers, batch size, features)
    print("LSTM Cell State Shape:", cell.shape)   # Shape: (num lstm layers, batch size, features)



def test_transformer():
    # Transformer expects input size (seq_len, batch, features)
    transformer = Transformer(feature_size=256, num_tokens=10000, num_heads=8, num_layers=6)
    dummy_input = torch.rand(10, 1, 256)  # Ex input tensor (sequence length, batch size, features)
    dummy_tgt = torch.randint(0, 10000, (1, 5))  # Target input for decoder
    output = transformer(dummy_input, dummy_tgt)
    print("Transformer Output Shape:", output.shape) # Shape: (sequence length, batch size, vocab size)



def test_lip_reading_model_dummy_data():
    model = LipReadingModel()
    # For CNN
    dummy_visual_input = torch.rand(1, 10, 3, 160, 160)  # (batch_size, num_frames, channels, height, width)
    # For transformer
    dummy_tgt = torch.randint(0, 1000, (1, 10))  # (seq_len, batch_size), num_tokens=1000 
    output = model(dummy_visual_input, dummy_tgt)
    print("Output shape:", output.shape)





def test_lip_reading_model_real_data():
    model = LipReadingModel()
    model.eval() 

    dataset = LipReadingDataset(directory='./LRS2/data_splits/val', transform=None) 
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for batch_idx, (frames, labels) in enumerate(data_loader):
        print("Frames shape:", frames.shape)  # Expected to be [batch_size, channels, height, width]
        print("Labels shape:", labels.shape)  # Expected to be [seq_len, batch_size] 
        
        output = model(frames, labels)
        print("Output shape:", output.shape)  

        break  # Only test one batch


if __name__ == "__main__":
    # # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    print("\nTesting dataloader...")
    test_data_loader()
    
    print("Testing CNN...")
    test_cnn()
    print("\nTesting LSTM...")
    test_lstm()
    print("\nTesting Transformer...")
    test_transformer()

    print("\nTesting lip reading on dummy data...")
    test_lip_reading_model_dummy_data()

    print("\nTesting lip reading on real data...")
    test_lip_reading_model_real_data()