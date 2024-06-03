# %%
from math import isnan
from torch.utils.data import DataLoader
from dataloader import LipReadingDataset
from speak import Speaking_conv3d_layers, Speaking_words, Speaking_model, Speaking_conv3d_layers_trimmed
import torch
import os
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch import nn
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
import torchvision
from transformers import AutoTokenizer
from main import validate

# %load_ext autoreload
# %autoreload 2

torch.manual_seed(42)



# # %%
# %%bash
# pwd
def main():
    base_path = "D:/classes/project/LRS2/extracted_data/mvlrs_v1/"
    dataset_name="pretrain"
    length_video=125

    device = "cuda"
    tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = Speaking_conv3d_layers_trimmed(512, 125)
    weights = torch.load("trimmed_shifted.state")["state_dict"]
    model.load_state_dict(weights)
    model.to(device)
    weight = torch.tensor([0.77, 0.23]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    train_dataset = LipReadingDataset(directory=base_path,
                            transform=None,
                            length_video=length_video,
                            mode="speak",
                            tokenizer=tokenizer,
                            dataset=dataset_name,
                            useOriginalDataStructure=True,
                            h_w = (160,160)

                            )
    print("Total samples loaded:", len(train_dataset))  
    train_set, val_set = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.9), len(train_dataset) - int(len(train_dataset)*0.9)])

    train_data_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=2
        
        )

    val_data_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=2,
        )
    print("Total samples train:", len(train_data_loader))  
    print("Total samples val:", len(val_data_loader))  
    validate(val_data_loader, model, criterion)

    # %%

    # params = []
    # i = 0
    # value = 0
    # increments = set(["4", "6", "8"])
    # new_layer = 4
    # for key, value in weights.items():
    #     if new_layer != 0:
    #         new_layer -= 1
    #         continue
    #     i += 1
    #     curr_val = name.split("encoder.")[1][0]
    #     if curr_val == str(i):
    #         new_layer = 3
    #         continue



    # for name, param in model.parameters():
    #     if "encoder" in name:
    #         curr_val = name.split("encoder.")[1][0]
    #         if curr_val in increments:
    #             increments.remove(curr_val)
    #             value += 1
    #         new_name = name.split("encoder.")
    #         new_val = str(int(curr_val) + value) 
    #         new_name = "encoder." + new_val + new_name[-1][1:]
    #         param.data = weights[new_name]
    #     else:
    #         param.data = weights[name]

    once = True
    values = 0
    new_weight3 = {}

    for key, value in list(new_weight2.items()):
        if "encoder" in key:
            curr_val = key.split("encoder.")[1][0]
            if once and curr_val == "5": 
                values = -1
                once = False
            elif curr_val == "5":
                continue
            new_name = key.split("encoder.")
            new_val = str(int(curr_val) + values) 
            if new_name[1][0:2] == "10":
                key = "encoder." + "9" + new_name[-1][2:]
            else:
                key = "encoder." + new_val + new_name[-1][1:]
        new_weight3[key] = value
            
        state = {
        
        'state_dict': new_weight3,
        }
        torch.save(state, "trimmed_shifted.state")


if __name__ == "__main__":
    main()
# %%

# %%



