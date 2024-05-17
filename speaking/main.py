from torchvision.datasets import VisionDataset

import torch
import argparse
import tensorboard
import numpy as np

def collate_fn(batch):
    sequences, labels = zip(*batch)
    encoded_labels = tokenizer(labels, add_special_tokens=True, max_length=100, padding="longest",  return_tensors='pt')
    
    return torch.unsqueeze(torch.stack(sequences[0]), axis=0), encoded_labels



def main(args):
    now = datetime.datetime.now()
    model = LipReadingModel()
    model.to(device)

    if args.train:
        save_path = f'{args.data_type}/Batch_size_{args.batch_size}/LR_{args.learning_rate}/Date_{now.month}_{now.day}_hr_{now.hour}'
        os.makedirs(save_path, exist_ok=True)
        writer = SummaryWriter(f'runs/{save_path}')

        train_dataset = LipReadingDataset(directory='./LRS2/data_splits/train' if os.getlogin() != "darke" else "D:/classes/cs231n/project/LRS2/data_splits/train", transform=None)
        print("Total samples loaded:", len(train_dataset))  
        data_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=False,
            )
        
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.99, 0.95), lr=args.learning_rate)
        # scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=int(len(train_dataset)/args.grad_accum_steps))
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Outputs: [batch_size, num_classes, sequence_length] Targets: [batch_size, sequence_length]

        for epoch in range(args.epochs):
            model.train() 
            total_loss = 0
            avg_loss = loss_accum = 0
            
            progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
            for batch_idx, (frames, targets) in progress_bar:
                frames, input_id, mask = frames.to(device, non_blocking=True), targets['input_ids'].to(device, non_blocking=True), targets['attention_mask'].to(device, non_blocking=True)
            
                output = model(frames, input_id, mask, args.train)

                loss = criterion(output, input_id) 
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

                writer.add_scalar('Training Loss', loss, epoch * len(data_loader) + batch_idx)
                writer.add_scalar('Average Batch Loss', avg_loss/(batch_idx+1), epoch * len(data_loader) + batch_idx)
                del loss, frames, mask, targets

                
                print("target \n",tokenizer.batch_decode(input_id))
                print("Guess \n",tokenizer.batch_decode(torch.argmax(output, dim=1)))
            average_loss = total_loss / len(data_loader)
            writer.add_scalar('Average Training Loss', average_loss, epoch)
            print(f"Average Loss for Epoch {epoch}: {average_loss}")
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
            torch.save(state, f"{epoch}.state")



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

    parser.add_argument('--learning_rate', default=0.001, type=int, help='learning rate for optimizer')
    # 3e-4 
    parser.add_argument('--epochs', default=2, type=int, help='num epoch to train for')
    args = parser.parse_args()
    main(args)