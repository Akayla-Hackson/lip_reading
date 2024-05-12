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
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor(),
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




if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_data_loader()



# model = LipReadingModel()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# for epoch in range(num_epochs):
#     for images, labels in dataloader:
#         optimizer.zero_grad()
#         output = model(images, labels)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()
