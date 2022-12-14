import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentiment import GloveLayer

glove_path = os.path.join(os.getcwd(), 'glove.6B.50d.txt')

class SentimentModel(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(SentimentModel, self).__init__()

        self.emb = GloveLayer(d=feature_size, glove_path=glove_path)
        self.lstm = nn.LSTM(input_size=feature_size, 
                            hidden_size=hidden_size, 
                            num_layers=4,
                            batch_first=True,
                            bidirectional=False)

    
    def forward(self, a, b):
        #a and b have shape (B, K). B batchsize, K num headlines in batch.
        #1. Split each headline into list of tokens
        #   (B, K) -> (B, K, L), L is maxlen of sequence
        #2. Embed each sequence
        #   (B, K, L) -> (B, K, L, D), D is 50 (glove vector)
        #3. LSTM
        #   (B, K, L, D) -> (B, K, L, D * H)
        

        
        emb_a = self.emb(a)
        emb_b = self.emb(b)

        out_a = self.lstm(emb_a)
        out_b = self.lstm(emb_b)



