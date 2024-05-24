import torchvision.models as models
from torch import nn
from classes.cnn import CNN
from classes.lstm import LSTM
import torch.nn.functional as F
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GptLipReadingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN() 
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, 768)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        # [84, 3, 90, 90]
        # print(x.size())
        cnn_out = self.cnn(x)    # Expected shape: (batch_size, updated num channels, updated height, updated width)
        cnn_out = cnn_out.view(batch_size, seq_len, -1) 
        # [1, 84, 512]
        # print(cnn_out.size())
        lstm_out, _ = self.lstm(cnn_out)   # Expected shape: (batch_size, sequence_length (num_frames), features)
        lstm_out = self.fc(lstm_out)  
        # [1, 84, 768]
        # print(lstm_out.size())
        
        #input_ids = self.gpt2_tokenizer.encode("CLS", return_tensors='pt')
        #input_ids = input_ids.to(lstm_out.device)

        #input_embeddings = self.gpt2_model.transformer.wte(input_ids)

        #print(input_ids.size())
        #print(lstm_out.size())
        #inputs_embeds = torch.cat((input_embeddings, lstm_out), dim=1)
        
       
        return lstm_out