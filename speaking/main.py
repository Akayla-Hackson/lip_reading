from math import isnan
from torch.utils.data import DataLoader
from dataloader import LipReadingDataset
from speak import Speaking_conv3d_layers, Speaking_words
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


torch.manual_seed(42)

def validate(loader, model, criterion, writer=None, total_len=0, idx=0, isWrite=False):
    progress_bar = tqdm(loader, total=len(loader))
    total_correct = total_words = 0
    zeros = ones = 0
    loss = 0
    poor = []
    if isWrite:
        file = open("./speaking/analyze_data/val_perf_lower_than50.txt", "w")
    for frames, label, video_path in progress_bar:
        frames, label, video_path = frames.to(device), label.to(device), video_path[0]
        with torch.no_grad():
            output = model(frames)
            correct = torch.sum(torch.argmax(output, axis=1) == label).detach().item()
            total =  label.shape[1]
            total_correct += correct
            total_words += total# this only works if its 1 batch size since other wise they will be padded

            if isWrite and correct/total <= 0.5:
                # file.write(f"{video_path} : {correct/total}\n")
                reader = torchvision.io.read_video(video_path, pts_unit = 'sec', output_format='TCHW')
                poor.append((correct/total,  reader[0].shape[0]/25, video_path))
            loss += criterion(output, label).detach().item() 
            ones += torch.sum(label==1)
            zeros += torch.sum(label==0)
    total = ones + zeros
    if isWrite:
        file.writelines(str(sorted(poor)))
        file.close()

    print(f"ones: {ones}, zeros: {zeros}, total: {total}, one_frac: {ones/total } zeors_frac: {zeros/total}")
    print(f"total_correct: {total_correct} total_words: {total_words}")
    print(f"accuracy: {total_correct/total_words}")
    print(f"loss: {loss}")

    if writer:
        writer.add_scalar('Val Accuracy', total_correct/total_words, total_len + idx)
        writer.add_scalar('loss Accuracy', loss, total_len + idx)



def main(args):
    length_video = 125 
    now = datetime.datetime.now()
    tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if args.useWords :
        model = Speaking_words(512, length_video, tokenizer.vocab_size)
        if args.train:
            checkpoint = torch.load(args.bestSpeakWeightsPath)  # use best weights of speaking
            weights = list(checkpoint['state_dict'].values())
            params = []
            i = 0
            for name, param in model.named_parameters():
                if "conv" in name and i < 32: 
                    param.data = weights[i]
                    i += 1
                else:
                    params.append(param)
            assert i == 32, f"not all weights were used only {i} out of 32"
    else:
        model = Speaking_conv3d_layers(512, length_video)
    model.to(device)
    weight = torch.tensor([0.77, 0.23]).to(device)
    if not args.useWords:
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=1)

    base_path = "D:/classes/project/" if args.useCloud != True else  "/home/jupyter/data/"

    train_dataset = LipReadingDataset(directory=base_path + "LRS2/data_splits/train",
                                transform=None,
                                length_video=length_video,
                                mode="word",
                                tokenizer=tokenizer)
    print("Total samples loaded:", len(train_dataset))  
    train_set, val_set = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.9), len(train_dataset) - int(len(train_dataset)*0.9)])

    train_data_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        )

    val_data_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        )
    print("Total samples train:", len(train_data_loader))  
    print("Total samples val:", len(val_data_loader))  


    if args.train:
        save_path = f'{args.data_type}/Batch_size_{args.batch_size}/LR_{args.learning_rate}/Date_{now.month}_{now.day}_hr_{now.hour}'
        os.makedirs(save_path, exist_ok=True)
        writer = SummaryWriter(f'runs_speak/{"words_" if args.useWords else ""}{save_path}')
        optimizer = torch.optim.AdamW(model.parameters() if args.useWords != True  else params, betas=(0.99, 0.95), lr=args.learning_rate)
        # scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=350, num_training_steps=int(len(train_data_loader)/args.grad_accum_steps))
  # Outputs: [batch_size, num_classes, sequence_length] Targets: [batch_size, sequence_length]

        if args.resume:
            print("Resuming training...")
            checkpoint = torch.load("0.state")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=int(len(train_data_loader)/args.grad_accum_steps), last_epoch=1)
 
        if args.compile:
            print("Compiling....")
            model = torch.compile(model,)
 
        for epoch in range(args.epochs):
            model.train() 
            total_loss = 0
            avg_loss = loss_accum = 0
            
            progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
            for batch_idx, (frames, label, _) in progress_bar:
                frames, label = frames.to(device, non_blocking=True), label.to(device, non_blocking=True)
            
                output = model(frames)

                loss = criterion(output, label) 
                if torch.isnan(loss):
                    loss = torch.nan_to_num(loss)
                loss_accum += loss
                loss = loss.detach()
                total_loss += loss
                avg_loss += loss

                if batch_idx != 0 and batch_idx % args.grad_accum_steps == 0:
                    # print(f"stepping....  loss: {loss_accum.detach().cpu}")
                    optimizer.zero_grad()
                    loss_accum.backward()
                    del loss_accum
                    nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                    loss_accum = 0
                    optimizer.step()
                    scheduler.step()
                    # if batch_idx % int(len(train_data_loader) * 0.2) == 0:
                    #     model.eval()
                    #     validate(val_data_loader, model, writer, epoch * len(train_data_loader), batch_idx)
                    #     model.train()

                writer.add_scalar('Learning rate ', torch.tensor(scheduler.get_last_lr()), epoch * len(train_data_loader) + batch_idx)
                writer.add_scalar('Training Loss', loss, epoch * len(train_data_loader) + batch_idx)

                writer.add_scalar('Average Batch Loss', avg_loss/(batch_idx+1), epoch * len(train_data_loader) + batch_idx)
                del loss, frames
                
            average_loss = total_loss / len(train_data_loader)
            writer.add_scalar('Average Training Loss', average_loss, epoch)
            print(f"Average Loss for Epoch {epoch}: {average_loss}")
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
            torch.save(state, f"{epoch}.state")
            model.eval()
            validate(val_data_loader, model, criterion, writer, epoch * len(train_data_loader), batch_idx)

    else:
        model.load_state_dict(torch.load(args.bestSpeakWeightsPath)['state_dict'])
        model.eval()
        validate(val_data_loader, model, criterion)




if __name__ == "__main__":
    # device = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type', default='train', type=str, help='dataset used for training')
    parser.add_argument('--batch_size', default=1, type=int, help='num entries per batch')
    parser.add_argument('--grad_accum_steps', default=16, type=int, help='How many steps to acumulate grad')
    parser.add_argument('--train', default=True, type=bool, help='Train or eval')


    parser.add_argument('--num_workers', default=4, type=int, help='num of workes for the dataloader')

    parser.add_argument('--learning_rate', default=6e-4, type=int, help='learning rate for optimizer')
    # 3e-4 
    parser.add_argument('--epochs', default=12, type=int, help='num epoch to train for')
    parser.add_argument('--resume', default=False, type=bool, help='resume training')
    parser.add_argument('--useWords', default=True, type=bool, help='train on words')
    parser.add_argument('--compile', default=False, type=bool, help='compile model')
    parser.add_argument('--useCloud', default=False, type=bool, help='Cloud compute')


    parser.add_argument('--bestSpeakWeightsPath', default="weights_trimed.state", type=str, help='train on words')


    
    args = parser.parse_args()
    main(args)