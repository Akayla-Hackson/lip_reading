import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2  # OpenCV for video processing
import re
import numpy as np
from PIL import Image

# Define the characters set
characters = list("abcdefghijklmnopqrstuvwxyz ")
# characters.append('-')  # CTC blank token
char_to_index = {char: idx for idx, char in enumerate(characters)}

class LRWDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.vocab_unordered = {}
        self.vocab_unordered[' '] = True
        for char in characters:
            self.vocab_unordered[char] = True

        self.samples = self._load_samples()
        self.vocab = []
        for char in self.vocab_unordered: self.vocab.append(char)
        self.vocab.sort()
        # invert ordered to create the char->int mapping
        # key: 1..N (reserve 0 for blank symbol)
        self.vocab_mapping = {}
        for i, char in enumerate(self.vocab):
            self.vocab_mapping[char] = i + 1
           
    def _load_samples(self):
        samples = []
        video_folders = sorted(os.listdir(self.directory))
        split_name = os.path.basename(self.directory)
        for video_folder in video_folders:
            label = os.path.splitext(video_folder)[0]
            label = ''.join(re.findall(r'[A-Za-z]', label))
            # build vocabulary
            for char in label.lower(): self.vocab_unordered[char] = True

            video_path = os.path.join(self.directory, video_folder)
            if not os.path.isdir(video_path):
                continue
            # print(video_path)
            sub_folders = sorted(os.listdir(video_path))

            for sub_folder in sub_folders:
                sub_folder_path = os.path.join(video_path, sub_folder)
                frames_dir = os.path.join(video_path, 'frames')
                if os.path.exists(frames_dir):
                    frames = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir)) if f.endswith('.jpg')]
                    samples.append((frames, label))
            if len(samples) > 1000:
                break        
        return samples            


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_path, label = self.samples[idx]
        frames = [self.transform(Image.open(frame)) for frame in frames_path]
        length = len(frames)
        frames = torch.stack(frames)

        y = []
        # allow whitespaces to be predicted
        for char in label.lower(): y.append(self.vocab_mapping[char])

        # label = torch.tensor([char_to_index[char] for char in label.lower()], dtype=torch.long)
        # return {'video': frames, 'label': label}
        
        return frames, y, length, idx