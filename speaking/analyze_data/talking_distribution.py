import os
import torchvision
import logging



def walk_directory(directory):
    video_folders = sorted(os.listdir(directory))
    split_name = os.path.basename(directory)


    for video_folder in video_folders:
        video_path = os.path.join(directory, video_folder)
        sub_folders = sorted(os.listdir(video_path))

        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(video_path, sub_folder)
            video_dir = os.path.join(sub_folder_path, sub_folder + '.mp4')
            label_file = os.path.join(sub_folder_path, sub_folder + '.txt')
                
            # print("Checking:", sub_folder_path)  

            if os.path.exists(label_file) and os.path.exists(video_dir):
                yield video_dir, label_file
                
def analyze_talk_no_talk(directory):
    talking = 0
    not_talking = 0
    for video_path, label_file in walk_directory(directory):
        with open(label_file, "r") as F:
            lines = F.read()
            lines = lines.splitlines()
            is_start_line = False
            start = 0
            end = 0
            for line in lines:
                if not is_start_line:
                    if line == "WORD START END ASDSCORE":
                        is_start_line = True
                    continue
                _, curr_start, curr_end, _ = line.split(" ")
                if curr_start != end: 
                    not_talking += 1
                elif curr_start == end:
                    talking += 1
                else:
                    logging.warning(f"start: {start} end: {end} not defined with curr_start: {curr_start} curr_end: {curr_end} file: {label_file}")
                
                start, end = curr_start, curr_end
    print(f"Talking: {talking} not talking: {not_talking} total: {talking+not_talking}\n")
    print(f"Talking fraction: {talking/(talking+not_talking)} not talking fraction: {not_talking/(talking+not_talking)}")
    with open(f"D:/classes/project/lip_reading/speaking/analyze_data/{directory.split("/")[-1]}_distribution", "w") as F:
        F.write(f"Talking: {talking} not talking: {not_talking} total: {talking+not_talking}\n")
        F.write(f"Talking fraction: {talking/(talking+not_talking)} not talking fraction: {not_talking/(talking+not_talking)}")


def analyze_frames_in_video(directory):
    longest_video = 0
    for video_path, label_file in walk_directory(directory):
        reader = torchvision.io.read_video(video_path, pts_unit = 'sec', output_format='TCHW')
        if reader[0].shape[0] > longest_video:
            longest_video = reader[0].shape[0]
            longest_path = video_path
    print(f"Longest video in frames: {longest_video} path: {longest_path}")
    with open(f"D:/classes/project/lip_reading/speaking/analyze_data/{directory.split("/")[-1]}_longest_video", "w") as F:
        F.write(f"Longest video in frames: {longest_video} path: {longest_path}")

    

for directory in ["D:/classes/project/LRS2/data_splits/train", "D:/classes/project/LRS2/data_splits/pretrain", "D:/classes/project/LRS2/data_splits/val"]:
    analyze_frames_in_video(directory)
    break

def poor_perfmance_parser():
    with open("./speaking/analyze_data/val_perf.txt", "r") as F:
        lines = F.read()
        lines = lines.splitlines()
        for line in lines:
            print()