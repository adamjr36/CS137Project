import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataloader_copy import MyDataset
import os
from googlenews import GoogleNews, getnews

root = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(root, 'cleaned_data')
csvs = os.listdir(data_dir)

class MyDataset2(MyDataset):
    def __init__(self, x, y, k=10, t=10, seed=None):
        super(MyDataset2, self).__init__(x, y, k, seed)

        self.t = t

    def __getitem__(self, idx):
        home, away, y = super().__getitem__(idx)

        x = self.x.iloc[idx].to_numpy()
        gnews = GoogleNews(language='en', max_results=10)
        homenews, awaynews = getnews(x, gnews)

        return home, homenews, away, awaynews, y

   