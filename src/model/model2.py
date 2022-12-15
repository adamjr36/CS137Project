import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentiment import GloveLayer
import numpy as np


class SentimentModel(nn.Module):

    def __init__(self, feature_size, hidden_size, glove_path):
        super(SentimentModel, self).__init__()
        self.maxlen = 30

        self.emb = GloveLayer(d=feature_size, 
                              glove_path=glove_path, 
                              max_length=self.maxlen)
        self.flat = nn.Flatten(0, 1)
        self.lstm = nn.LSTM(input_size=feature_size, 
                            hidden_size=hidden_size, 
                            num_layers=4,
                            batch_first=True,
                            bidirectional=False)


    def forward(self, a, b):
        #a and b have shape (N, K). N batchsize, K num headlines in batch.
        #1. Split each headline into list of tokens
        #   (N, K) -> (N, K, L), L is maxlen of sequence

        N, K = a.shape
        L = self.maxlen
        D = 50

        for headlines in a:
            for i, seq in enumerate(headlines):
                headlines[i] = np.array(seq.split()) 
        for headlines in b:
            for i, seq in enumerate(headlines):
                headlines[i] = np.array(seq.split()) 


        #2. Embed each sequence
        #   (N, K, L) -> (N, K, L, D), D is 50 (glove vector)
        emb_a = torch.zeros((N, K, L, D))
        emb_b = torch.zeros((N, K, L, D))
        for i in range(N):
            emb_a[i] = torch.tensor(self.emb(a[i]))
            emb_b[i] = torch.tensor(self.emb(b[i]))

        #3. Prepare for LSTM.
        #Flatten first two dims to get (N, K, L, D) -> (N x K, L, D)
        print(emb_a.shape)
        emb_a = self.flat(emb_a)
        emb_b = self.flat(emb_b)

        #3. LSTM
        #   (N x K, L, D) -> (N x K, L, H) (hn)

        outa, (hna, cna) = self.lstm(emb_a)
        outb, (hnb, cnb) = self.lstm(emb_b)

        #4. Retrive Hidden layer onlly at last output step (eliminate L)

        outa = outa.select(1, -1)
        outb = outb.select(1, -1)

        #5. Reshape (N x K, H) -> (N, K x H)

        outa = outa.reshape(N, -1)
        outb = outb.reshape(N, -1)

        return outa, outb

