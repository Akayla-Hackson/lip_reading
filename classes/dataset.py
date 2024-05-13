import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

class LipReadingDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directory with all the video folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform or transforms.ToTensor()
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        video_folders = sorted(os.listdir(self.directory))
        split_name = os.path.basename(self.directory)
        for video_folder in video_folders:
            video_path = os.path.join(self.directory, video_folder)
            sub_folders = sorted(os.listdir(video_path))

            for sub_folder in tqdm(sub_folders, desc=f"Loading dataset: {split_name}", leave=False):
                sub_folder_path = os.path.join(video_path, sub_folder)
                frames_dir = os.path.join(sub_folder_path, 'frames')
                label_file = os.path.join(sub_folder_path, sub_folder + '.txt')
                 
                # print("Checking:", sub_folder_path)  # Debug

                if os.path.exists(label_file) and os.path.isdir(frames_dir):
                    label = self.read_phrase_from_file(label_file)
                    frames = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir)) if f.endswith('.jpg')]
                    samples.append((frames, label))
                else:
                    print("Skipped:", sub_folder_path)  # Debug

        return samples



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_path, label = self.samples[idx]
        frames = [self.transform(Image.open(frame)) for frame in frames_path]
        return frames, label 

    def read_phrase_from_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        phrase = ""
        for line in lines:
            if "Text:" in line:
                parts = line.split("Text:")
                if len(parts) > 1:
                    phrase = parts[1]. strip()
                    break 
        return phrase
