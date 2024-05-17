import torchvision
import os
import itertools
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms as t

video_path = "path_to_a_test_video"
reader = torchvision.io.VideoReader(video_path, "video")
reader.seek(2.0)
frame = next(reader)


reader.seek(2)
for frame in reader:
    frames.append(frame['data'])
# additionally, `seek` implements a fluent API, so we can do
for frame in reader.seek(2):
    frames.append(frame['data'])


# and similarly, reading 10 frames after the 2s timestamp can be achieved as follows:
    for frame in itertools.islice(reader.seek(2), 10):
    frames.append(frame['data'])

class LipReadingDataset(Dataset):
    def __init__(self, directory, transform=None, resolution=0.5):
        """
        Args:
            directory (string): Directory with all the video folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        transform = [t.ToTensor(), t.Resize((int(180*resolution), int(180*resolution))), ]  # 90x90
        self.transform  = t.Compose(transform)
        self.samples = self._load_samples()
        self.resolution = resolution

    def _load_samples(self):
        samples = []
        video_folders = sorted(os.listdir(self.directory))
        split_name = os.path.basename(self.directory)
        for video_folder in video_folders:
            video_path = os.path.join(self.directory, video_folder)
            sub_folders = sorted(os.listdir(video_path))

            for sub_folder in sub_folders:
                sub_folder_path = os.path.join(video_path, sub_folder)
                video_dir = os.path.join(sub_folder_path, sub_folder + '.mp4')
                label_file = os.path.join(sub_folder_path, sub_folder + '.txt')
                 
                # print("Checking:", sub_folder_path)  

                if os.path.exists(label_file) and os.path.isdir(video_dir):
                    label = self.read_phrase_from_file(label_file)
                    frames = [os.path.join(video_dir, f) for f in sorted(os.listdir(video_dir)) if f.endswith('.jpg')]
                    samples.append((frames, label))
                # else:
                #     print("Skipped:", sub_folder_path)  

            # if os.getlogin() == "darke" and len(samples) > 1000:
            # if len(samples) > 100:
            #     break

        return samples
