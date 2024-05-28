import torchvision
import os
import itertools
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms as t
import numpy as np
import math
import torch

def load_samples(directory):
    samples = []
    video_folders = sorted(os.listdir(directory))
    split_name = os.path.basename(directory)
    for video_folder in video_folders:
        video_path = os.path.join(directory, video_folder)
        sub_folders = sorted(os.listdir(video_path))

        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(video_path, sub_folder)

            video_dir = os.path.join(sub_folder_path, sub_folder + '.mp4')
            label_file = os.path.join(sub_folder_path, sub_folder + '.txt')

            if os.path.exists(label_file) and os.path.exists(video_dir):
                samples.append((video_dir, label_file))
    return samples

def get_file_data(F):
        lines = F.read()
        lines = lines.splitlines()
        is_start_line = False
        for line in lines:
            if not is_start_line:
                if line == "WORD START END ASDSCORE":
                    is_start_line = True
                continue
            yield  line.split(" ")

def get_frames(length_video, video_path, transform, h_w):
    reader = torchvision.io.read_video(video_path, pts_unit = 'sec', output_format='TCHW')
    c,h,w = reader[0].shape[1:]
    h, w = h_w
    frames = torch.zeros((length_video, c, h, w), dtype=torch.float32)
    # frames_path, label = self.samples[0]
    for i, frame in enumerate(reader[0]):
        frames[i] = transform(frame)
        if i == length_video -1:
            break
    frames = frames.permute(1,0,2,3)
    return frames, reader[-1]["video_fps"]

class LipReadingDataset(Dataset):
    def __init__(self, directory, transform=None, resolution=0.5, length_video=200, mode="word", tokenizer=None, h_w = (96,96)):
        """
        Args:
            directory (string): Directory with all the video folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        transform = [t.Resize(h_w), t.ToPILImage(), t.ToTensor()]  # 90x90
        self.transform  = t.Compose(transform)
        self.samples = load_samples(directory)
        self.resolution = resolution
        self.length_video = length_video
        self.mode = mode
        self.tokenizer = tokenizer
        self.h_w = h_w
        if mode not in ["speak", "word"]:
            raise NotImplementedError( f"{self.mode} was enterered as mode in dataloader.  this is not supported")
       
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames, frame_rate = get_frames(self.length_video, video_path, self.transform, self.h_w )
        labels_np = torch.zeros((self.length_video)) if self.mode != "word" else torch.ones((self.length_video)) 
        labels_np = self.get_labels(label, labels_np,  frame_rate)
        return frames, labels_np, video_path
    
    def get_labels(self, label_file, labels_np, frame_rate):
        with open(label_file, "r") as F:
            for word, curr_start, curr_end, _ in get_file_data(F):
                curr_start = math.ceil(float(curr_start) * frame_rate)
                curr_end = math.floor(float(curr_end) * frame_rate)
                if self.mode != "word":
                    labels_np[curr_start: curr_end + 1] = 1
                else:
                    labels_np[curr_start: curr_end + 1] = self.tokenizer(word, add_special_tokens=False)['input_ids'][0]
                if curr_end == self.length_video:
                    break
        return labels_np.type(torch.LongTensor) 



# if __name__ == "__main__":
#     train_dataset = LipReadingDataset(directory='./LRS2/data_splits/train' if os.getlogin() != "darke" else "D:/classes/project/LRS2/data_splits/train", transform=None)
#     data_loader = DataLoader(
#                 train_dataset,
#                 batch_size=1,
#                 shuffle=True,
#                 # collate_fn=collate_fn,
#                 num_workers=1,
#                 pin_memory=False,
#                 )
#     for batch_idx, (frames, targets) in enumerate(data_loader):
#         print("")
        