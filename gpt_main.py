import torch
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from classes.dataset import LipReadingDataset
from classes.cnn import CNN
from classes.gpt_lstm import LSTM
from classes.transformer import Transformer
from classes.lip_reading import LipReadingModel
from classes.hw_lip_reading import HwLipReadingModel
from classes.gpt_lip_reading import GptLipReadingModel
from classes.hw_transformer import *
from classes.hw_transformer_layers import *
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
import datetime
import numpy as np
import random
from torch import nn
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from jiwer import wer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import math

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
# config.add_cross_attention = True
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    # labels = [item for item in labels]
    # encodings = gpt2_tokenizer.batch_encode_plus(
    #     labels,
    #     padding=True,      
    #     truncation=True,   
    #     return_tensors='pt')
    print(labels[0])
    encodings = gpt2_tokenizer.encode(labels[0], return_tensors='pt')
    
    # return torch.unsqueeze(torch.stack(sequences[0]), axis=0), encodings['input_ids'], encodings['attention_mask'], labels
    return torch.unsqueeze(torch.stack(sequences[0]), axis=0), encodings, labels[0]


def main(args):
    now = datetime.datetime.now()
    lstm_model = LSTM()
    model = GptLipReadingModel(lstm_model)
    for parameter in model.parameters():        
        parameter.requires_grad = True
    model.to(device)
    gpt2_model = GPT2LMHeadModel(config).to(device)
    for parameter in gpt2_model.lm_head.parameters():        
        parameter.requires_grad = True
    if args.train:
        save_path = f'{args.data_type}/Batch_size_{args.batch_size}/LR_{args.learning_rate}/Date_{now.month}_{now.day}_hr_{now.hour}'
        os.makedirs(save_path, exist_ok=True)
        writer = SummaryWriter(f'runs/{save_path}')

        train_dataset = LipReadingDataset(directory='./LRS2/data_splits/pretrain' if os.getlogin() != "darke" else "D:/classes/cs231n/project/LRS2/data_splits/train", transform=None)
        print("Total samples loaded:", len(train_dataset))  
        data_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()  # Outputs: [batch_size, num_classes, sequence_length] Targets: [batch_size, sequence_length]
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=int(len(train_dataset)/args.grad_accum_steps))
        loss_history = []
        for epoch in range(args.epochs):
            model.train() 
            gpt2_model.train() 
           # optimizer.zero_grad()

            total_loss = 0 
            avg_loss = loss_accum = 0
            total_wer = 0.0
            progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
            # for batch_idx, (frames, targets, masks, labels) in progress_bar:
            for batch_idx, (frames, targets, labels) in progress_bar:
                # targets: {'input_ids': tensor([[ 101, 2009, 1005, 1055, 2183, 2000, 2175, 2006, 2046, 1996, 2925,  102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
                # frames, input_id, masks = frames.to(device, non_blocking=True), targets['input_ids'].to(device, non_blocking=True), targets['attention_mask'].bool().to(device, non_blocking=True)
                frames = frames.to(device, non_blocking=True)
                targets = targets.to(device)
                # masks = masks.to(device)
                # frames.size: torch.Size([1, 84, 3, 90, 90])
                # print(frames.size())
                
                # 11
                # print(len(labels.split()))

                # [1, 20]
                # print(targets.size())
                
                # embeds = model(frames)
                # print(embeds.requires_grad)
                
                input_ids = gpt2_tokenizer.encode("Transcribe the spoken words: ", return_tensors='pt').to(device)
                # input_length = input_ids.shape[1]
                # [1, 8]
                # print(input_ids.size())

                # outputs = gpt2_model.generate(inputs_embeds=embeds, max_new_tokens=targets.size(1), 
                    #do_sample=True, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, 
                #     output_scores=True, return_dict_in_generate=True)
                # same size
                # logits = torch.stack(outputs.scores, dim=1)
                #print(logits.size())
                #print(len(outputs.logits))
                #print(targets.size())
                
                # output = gpt2_model.generate.__wrapped__(gpt2_model, inputs_embeds=embeds, output_scores=True, 
                #     return_dict_in_generate=True,
                #     max_new_tokens=targets.size(1), do_sample=True, num_beams=5, no_repeat_ngram_size=2, early_stopping=True) 
                # print(output.scores.size())
                
                # outputs = gpt2_model(inputs_embeds=embeds, labels=targets)
                # [1, 192, 50257]
                # print(outputs.logits.size())
                # loss = outputs.loss
                
                # loss = criterion(logits.view(-1, 50257), targets.view(-1))

                embeds = model(frames, targets.size(1))
                outputs = gpt2_model(inputs_embeds=embeds, labels=targets)
                logits = outputs.logits
                # print(outputs.logits.size())
                
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                


                loss_accum += loss
                loss = loss.detach()
                total_loss += loss
                avg_loss += loss

                if batch_idx != 0 and batch_idx % args.grad_accum_steps == 0:
                    optimizer.zero_grad()
                    loss_accum.backward()
                    del loss_accum
                    nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                    loss_accum = 0
                    optimizer.step()
                    scheduler.step()

                # loss.backward()
                # optimizer.step()
                # total_loss += loss.item()

                writer.add_scalar('Training Loss', loss.item(), epoch * len(data_loader) + batch_idx)

                # Generate text using GPT-2 with inputs_embeds
                # generated_ids = gpt2_model.generate(inputs_embeds=embeds)
                # predicted_text = gpt2_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                # Get the predicted token IDs
                predicted_token_ids = torch.argmax(logits, dim=-1)

                # Decode the predicted token IDs to text
                predicted_text = gpt2_tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

                print("target \n", labels)
                print("Guess \n", predicted_text)
                sample_wer = wer(labels, predicted_text)
                total_wer += sample_wer
            average_loss = total_loss / len(data_loader)
            writer.add_scalar('Average Training Loss', average_loss, epoch)
            print(f"Average Loss for Epoch {epoch}: {average_loss}")

            # Calc WER for each item in the batch and accumulate
            average_wer = total_wer / len(data_loader)
            print(f"Average WER: {average_wer:.2f}") 

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(trainable_params)

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
            torch.save(state, f"{epoch}.state")
    else:
        save_path = f'Val/{args.data_type}/Batch_size_{args.batch_size}/LR_{args.learning_rate}/Date_{now.month}_{now.day}_hr_{now.hour}'
        os.makedirs(save_path, exist_ok=True)
        writer = SummaryWriter(f'runs/{save_path}')

        model.load_state_dict(torch.load("0.state")['state_dict'])
        model.eval()
        # print(model)

        val_dataset = LipReadingDataset(directory='./LRS2/data_splits/val' if os.getlogin() != "darke" else "D:/classes/cs231n/project/LRS2/data_splits/val", transform=None)
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=False,
        )

        total_words = 0
        total_wer = 0.0
        num_samples = 0
        print("Total samples loaded for validation:", len(val_data_loader))  

        for batch_idx, (frames, targets) in enumerate(val_data_loader):
            with torch.no_grad():
                frames, input_id, mask = frames.to(device, non_blocking=True), targets['input_ids'].to(device, non_blocking=True), targets['attention_mask'].to(device, non_blocking=True)
                print("target \n",tokenizer.batch_decode(input_id))
                output = model(frames, input_id, args.train)
                output = output.argmax(axis=1)
                
                predicted_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
                reference_texts = tokenizer.batch_decode(input_id, skip_special_tokens=True)
                
                # Calc WER for each item in the batch and accumulate
                for predicted, reference in zip(predicted_texts, reference_texts):
                    sample_wer = wer(reference, predicted)
                    total_wer += sample_wer
                    num_samples += 1
                # print("Predictions:", predicted_texts)
                # print("References:", reference_texts)

            print("target \n",tokenizer.batch_decode(input_id))
            print("Guess \n",tokenizer.batch_decode(output))
            print("WER:", sample_wer)
            writer.add_scalar('Val WER', sample_wer, len(val_data_loader) + batch_idx)
        
        # Calc avg WER across all samples
        average_wer = total_wer / num_samples
        print(f"Average WER: {average_wer:.2f}")
        writer.add_scalar('Average WER', average_wer)

if __name__ == "__main__":
    # device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():      
        device = 'mps'                         
    else:
        device = torch.device('cpu')
    print(device)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_type', default='train', type=str, help='dataset used for training')
    parser.add_argument('--batch_size', default=1, type=int, help='num entries per batch')
    parser.add_argument('--grad_accum_steps', default=16, type=int, help='How many steps to acumulate grad')
    parser.add_argument('--train', default=True, type=bool, help='Train or eval')


    parser.add_argument('--num_workers', default=4, type=int, help='num of workes for the dataloader')

    parser.add_argument('--learning_rate', default=0.001, type=int, help='learning rate for optimizer')
    # 3e-4 
    parser.add_argument('--epochs', default=10, type=int, help='num epoch to train for')
    args = parser.parse_args()
    main(args)