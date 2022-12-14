import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from googlenews import GoogleNews, getnews

def make_inputs():
    df = pd.read_csv(os.path.join(data_dir, 'master_data.csv'))
    df = df[['Match', 'Home Team', 'Away Team', 'Date', 'Win', 'Team', 'Home']]
    df = df.drop_duplicates(subset=['Match'])
    
    #Convert 'Win' to 'Home Win'
    df['Home'] = (df['Home'] * 2) - 1
    df['Win'] = df['Home'] * df['Win']
    #

    # Home team, Away Team, Date, Win, ...

    df_x = df[['Home Team', 'Away Team', 'Date']]
    df_y = df[['Win']]

    df_x.to_csv(os.path.join(data_dir, 'x.csv'))
    df_y.to_csv(os.path.join(data_dir, 'y.csv'))


def match_history(data_dir, team, date, k):
    df = team_df(data_dir, team)
    df = df[df['Date'] < date]
    df = df.sort_values('Date', ascending=False).head(k)
    return df #drop cols

def team_df(data_dir, team):
    df = pd.read_csv(os.path.join(data_dir, '{}_data.csv'.format(team)))
    return df


class MyDataset(Dataset):
    def __init__(self, data_dir, x, y, k=10, seed=None):
        self.data_dir = data_dir
        x = pd.read_csv(os.path.join(data_dir, x))
        y = pd.read_csv(os.path.join(data_dir, y))
        x['Win'] = y['Win']
        x = x.sample(frac=1, random_state=seed).reset_index()

        self.x = x[['Home Team', 'Away Team', 'Date']]
        self.y = x['Win']

        assert(self.x.shape[0] == self.y.shape[0])
        self.k = k

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        x = self.x.iloc[idx]
        y = self.y.iloc[idx]

        single = type(idx) == int
        if (single):
            y = np.array(y)
        else:
            y = y.to_numpy()
        
        #[0] is home, [1] away, [2] date
        #first dim might be multiple games
        x = x.to_numpy()
            
        N = 1 if single else x.shape[0]
        K = self.k
        D = 110

        homearray = np.full((N, K, D), np.nan, dtype=object)
        awayarray = np.full((N, K, D), np.nan, dtype=object)

        def get_data(data_dir, row, i):
            home = row[0]
            away = row[1]
            date = row[2]

            homedf = match_history(data_dir, home, date, self.k).to_numpy()
            awaydf = match_history(data_dir, away, date, self.k).to_numpy()

            # print(homedf.shape)

            homedf = np.pad(homedf, [(0, self.k - len(homedf)), (0, 0)])
            awaydf = np.pad(awaydf, [(0, self.k - len(awaydf)), (0, 0)])

            homearray[i] = homedf
            awayarray[i] = awaydf

        if not single:
            for i in range(N):
                get_data(self.data_dir, x[i], i)
        else:
            get_data(self.data_dir, x, 0)

        homearray = np.delete(homearray, [0, 1, 2, 3], axis=2)
        awayarray = np.delete(awayarray, [0, 1, 2, 3], axis=2)

        #homearray = np.squeeze(homearray, axis=0)
        #awayarray = np.squeeze(awayarray, axis=0)

        return np.array(homearray, dtype=np.float32), np.array(awayarray, dtype=np.float32), y


class MyDataset2(MyDataset):
    def __init__(self, data_dir, x, y, k=10, t=10, seed=None):
        super(MyDataset2, self).__init__(data_dir, x, y, k, seed)

        self.t = t

    def __getitem__(self, idx):
        home, away, y = super().__getitem__(idx)

        x = self.x.iloc[idx].to_numpy()
        gnews = GoogleNews(language='en', max_results=self.t)
        homenews, awaynews = getnews(x, gnews)

        return home, homenews, away, awaynews, y


if __name__ == '__main__':

    #for csv in csvs:
     #   df = pd.read_csv(os.path.join(data_dir, csv))
      #  print(csv)
       # print(len(df.columns))
    
    dataset = MyDataset('x.csv', 'y.csv')
    
    
    x1, x2, y = dataset[0]
    # print(dataset.x.iloc[0])
    # print(x1.shape, x2.shape)
    print(x1, x2, y)