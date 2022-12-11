import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os

root = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(root, 'cleaned_data')
csvs = os.listdir(data_dir)

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


def match_history(team, date, k):
    df = team_df(team)
    df = df[df['Date'] < date]
    df = df.sort_values('Date', ascending=False).head(k)
    return df #drop cols

def team_df(team):
    df = pd.read_csv(os.path.join(data_dir, '{}_data.csv'.format(team)))
    return df


class MyDataset(Dataset):
    def __init__(self, x, y, k=10, seed=None):
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
        D = 112

        homearray = np.full((N, K, D), np.nan, dtype=object)
        awayarray = np.full((N, K, D), np.nan, dtype=object)

        def get_data(row, i):
            home = row[0]
            away = row[1]
            date = row[2]

            homedf = match_history(home, date, self.k).to_numpy()
            awaydf = match_history(away, date, self.k).to_numpy()

            print(homedf.shape)

            homedf = np.pad(homedf, [(0, self.k - len(homedf)), (0, 0)])
            awaydf = np.pad(awaydf, [(0, self.k - len(awaydf)), (0, 0)])

            homearray[i] = homedf
            awayarray[i] = awaydf

        if not single:
            for i in range(N):
                get_data(x[i], i)
        else:
            get_data(x, 0)

        return homearray, awayarray, y

if __name__ == '__main__':

    #for csv in csvs:
     #   df = pd.read_csv(os.path.join(data_dir, csv))
      #  print(csv)
       # print(len(df.columns))
    
    dataset = MyDataset('x.csv', 'y.csv')
    
    
    x1, x2, y = dataset[0]
    print(dataset.x.iloc[0])
    print(x1.shape, x2.shape)
    print(x1, x2, y)