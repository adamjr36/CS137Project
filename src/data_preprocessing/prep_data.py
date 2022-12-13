import pandas as pd
import os
from sklearn.model_selection import train_test_split

root = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(root, 'cleaned_data')

def tt_split(seed=None, test_size=.2):
    df_x = pd.read_csv(os.path.join(data_dir, 'x.csv'))
    df_y = pd.read_csv(os.path.join(data_dir, 'y.csv'))

    df_x['Win'] = df_y['Win']

    x = x.sample(frac=1, random_state=seed).reset_index()
    train, test = train_test_split(x, test_size=test_size)

    trainy = train['Win']
    trainx = train[['Home Team', 'Away Team', 'Date']]
    
    testy = test['Win']
    testx = test[['Home Team', 'Away Team', 'Date']]

    trainx.to_csv(os.path.join(data_dir, 'x_train.csv'))
    trainy.to_csv(os.path.join(data_dir, 'y_train.csv'))
    testx.to_csv(os.path.join(data_dir, 'x_test.csv'))
    testy.to_csv(os.path.join(data_dir, 'y_test.csv'))
    
if __name__ == '__main__':
    tt_split()

