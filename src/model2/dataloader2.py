import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataloader_copy import MyDataset
import os
from googlenews import GoogleNews, getnews


class MyDataset2(MyDataset):
    def __init__(self, data_dir, x, y, k=10, t=10, seed=None):
        super(MyDataset2, self).__init__(data_dir, x, y, k, seed)

        self.t = t

    def __getitem__(self, idx):
        home, away, y = super().__getitem__(idx)

        x = self.x.iloc[idx].to_numpy()
        gnews = GoogleNews(language='en', max_results=10)
        homenews, awaynews = getnews(x, gnews)

        '''for news in homenews:
            for i, n in enumerate(news):
                news[i] = n['title']
        for news in awaynews:
            for i, n in enumerate(news):
                news[i] = n['title']'''

        return home, homenews, away, awaynews, y

   