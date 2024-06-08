import torchvision.models as models
from torch import nn
from classes.cnn import CNN
from classes.gpt_lstm import LSTM
import torch.nn.functional as F
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model


class GptLipReadingModel(nn.Module):
    def __init__(self, lstm_model):
        super().__init__()
        self.cnn = CNN() 
        self.lstm = lstm_model
        self.relu = nn.ReLU()

        self.fc = nn.Linear(256, 768)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.gpt = GPT2Model.from_pretrained('gpt2')
        self.linear = nn.Linear(512, 768)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt.config.is_decoder = True

    def forward(self, x, target_length):
        
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        # [84, 3, 90, 90]
        # print(x.size())
        cnn_out = self.cnn(x)    # Expected shape: (batch_size, updated num channels, updated height, updated width)
        # [84, 512, 1, 1]
        # print(cnn_out.size())
        
        cnn_out = cnn_out.view(batch_size, seq_len, -1) 

        # [1, 84, 512]
        # print(cnn_out.size())
        lstm_out = self.lstm(cnn_out, target_length)   # Expected shape: (batch_size, sequence_length (num_frames), features)
        # out = self.fc(lstm_out)
        # print(lstm_out.size()) 
        
        return lstm_out

        lstm_out = lstm_out.permute(0, 2, 1)  # Permute for pooling
        pooled_out = self.pooling(lstm_out)  # pooled_out: [batch_size, hidden_size, reduced_seq_len]
        pooled_out = pooled_out.permute(0, 2, 1)

        out = self.fc(pooler_out)

        return out
        # [1, 84, 512]
        # print(cnn_out.size())
        # lstm_out, _ = self.lstm(cnn_out)   # Expected shape: (batch_size, sequence_length (num_frames), features)
        # embeddings = self.linear(cnn_out)  
        # output = output.permute(0, 2, 1)  # Permute for pooling

        # out = self.pooling(lstm_out)
        # [1, 84, 768]
        # print(lstm_out.size())
        
        #input_ids = self.gpt2_tokenizer.encode("CLS", return_tensors='pt')
        #input_ids = input_ids.to(lstm_out.device)

        #input_embeddings = self.gpt2_model.transformer.wte(input_ids)

        #print(input_ids.size())
        #print(lstm_out.size())
        #inputs_embeds = torch.cat((input_embeddings, lstm_out), dim=1)
        #outputs = self.gpt(inputs_embeds=embeddings)
        # return outputs.last_hidden_state
       
        # return lstm_out

class VideoProcessingModel(nn.Module):
    def __init__(self, conv3d_model, lstm_model):
        super().__init__()
        self.conv3d_model = conv3d_model
        self.lstm_model = lstm_model

    def forward(self, x, target_length):
        # x: [batch_size, channels, depth, height, width]
        features = self.conv3d_model(x)  # [batch_size, out_channels, depth, height, width]
        batch_size, channels, depth, height, width = features.size()
        features = features.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch_size, seq_len, input_size]
        output = self.lstm_model(features, target_length)  # [batch_size, target_length, hidden_size]
        return output        