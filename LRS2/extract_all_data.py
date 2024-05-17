import os
import tarfile
import shutil
import cv2
<<<<<<< HEAD
import re 
=======
>>>>>>> main

def concatenate_parts(output_dir, base_filename, parts):
    full_tar_path = os.path.join(output_dir, base_filename + '.tar')
    with open(full_tar_path, 'wb') as full_tar:
        for part in sorted(parts):
            part_path = os.path.join(output_dir, part)
            with open(part_path, 'rb') as file_part:
                shutil.copyfileobj(file_part, full_tar)
    print(f"All parts concatenated into {full_tar_path}")
    return full_tar_path


# Extract tar file
def extract_tar(tar_path, extract_to):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted tar file to {extract_to}")


# Create the dataset splits 
def move_files_based_on_txt(extracted_directory, dataset_names, data_splits_dir, root_path):

    if not os.path.exists(data_splits_dir):
        os.makedirs(data_splits_dir)

    # Directories to search within extracted data
    search_dirs = ['main', 'pretrain']

    # Process each .txt file for dataset directories
    for dataset in dataset_names:
        txt_file_path = os.path.join(root_path, dataset + '.txt')
        print(f"Processing {txt_file_path}...")
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as file:
                for line in file:
                    video_id = line.strip().split()[0]
                    new_folder_path = os.path.join(data_splits_dir, dataset, video_id)

                    if not os.path.exists(new_folder_path):
                        os.makedirs(new_folder_path)
                            
                    # Check in both 'main' and 'pretrain' directories
                    found = False
                    for search_dir in search_dirs:
                        video_path = os.path.join(extracted_directory, search_dir, video_id + '.mp4')
                        text_path = os.path.join(extracted_directory, search_dir, video_id + '.txt')

                        if os.path.exists(video_path) and os.path.exists(text_path):
                            shutil.move(video_path, new_folder_path)
                            shutil.move(text_path, new_folder_path)
                            print(f"Moved {video_id}.mp4 and {video_id}.txt to {new_folder_path}")
                            found = True
                            break
                    if not found:
                        print(f"Files for {video_id} not found in any directory.")
        else:
            print(f"File {txt_file_path} does not exist")
    
    shutil.rmtree(extracted_directory)

def lrw_move_files_based_on_txt(extracted_directory, dataset_names, data_splits_dir, root_path):

    if not os.path.exists(data_splits_dir):
        os.makedirs(data_splits_dir)

    extract_path = './LRS2/lipread_mp4'

    # Process each .txt file for dataset directories
    for word in os.listdir(extract_path):
        word_dir_path = os.path.join(extract_path, word)
        for dataset in os.listdir(word_dir_path):
            dataset_dir_path = os.path.join(word_dir_path, dataset)
            for sample in os.listdir(dataset_dir_path):
                video_id = os.path.splitext(sample)[0]
                sampel_path = os.path.join(dataset_dir_path, sample)
                new_folder_path = os.path.join(data_splits_dir, dataset, video_id)
                if not os.path.exists(new_folder_path):
                        os.makedirs(new_folder_path)

                shutil.move(sampel_path, new_folder_path)
                
                print(f"Moved {video_id}.mp4 and {video_id}.txt to {new_folder_path}")

    

# Get frames from mp4 & save in directory (called in process_datasets)
def extract_frames(video_path, frames_dir):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    # fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    # print(fps)

    while success:
        cv2.imwrite(os.path.join(frames_dir, f"frame{count}.jpg"), image)  # save frame as jpg
        success, image = vidcap.read()
        count += 1
    vidcap.release()

# Moves thru directories to create the frames
def process_datasets(root_dir, dataset_names):
    for dataset in dataset_names:
        dataset_path = os.path.join(root_dir, dataset)
        if os.path.exists(dataset_path):
            for video_id_dir in os.listdir(dataset_path):
                video_id_path = os.path.join(dataset_path, video_id_dir)
                if os.path.isdir(video_id_path):
                    for number_dir in os.listdir(video_id_path):
                        number_dir_path = os.path.join(video_id_path, number_dir)
                        if number_dir.endswith('.mp4'):
                            frames_dir = video_id_path + '/frames'
                            extract_frames(number_dir_path, frames_dir)
                        # if os.path.isdir(number_dir_path):
                        #     for item in os.listdir(number_dir_path):
                        #         if item.endswith('.mp4'):
                        #             video_path = os.path.join(number_dir_path, item)
                        #             frames_dir = number_dir_path + '/frames'
                        #             print(f"Extracting frames from {video_path} to {frames_dir}")
                        #             extract_frames(video_path, frames_dir)
                                   
# Paths & params
extract_to = './LRS2/extracted_data'  
output_directory = '.'  
#dataset_names = ['pretrain', 'train', 'val', 'test']
dataset_names = ['train', 'val', 'test']

base_filename = 'lrw-v1-partaa'
parts = ['lrw-v1-partaa']
extracted_directory = './LRS2/extracted_data/mvlrs_v1'
data_splits_dir = './LRS2/data_splits'
root_path = './LRS2/'

# tar_file_path = './LRS2/lrs2_v1.tar' 

# tar_file_path = concatenate_parts(output_directory, base_filename, parts)
# extract_tar(tar_file_path, extract_to)
lrw_move_files_based_on_txt(extracted_directory, dataset_names, data_splits_dir, root_path)
process_datasets(data_splits_dir, dataset_names)