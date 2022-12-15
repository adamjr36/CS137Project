import pandas as pd
import os

root = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(root, 'cleaned_data')

csvs = os.listdir(data_dir)

#def train_test_split():
   
#Chronological
def train_test_split():
    dfx = pd.read_csv(os.path.join(data_dir, 'x.csv'))
    dfy = pd.read_csv(os.path.join(data_dir, 'y.csv'))

    dfx['Win'] = dfy['Win']

    dfx.sort_values('Date')

    n = len(dfx)
    trva = int(.8 * n)
    te = n - trva

    df_test = dfx.tail(te)
    
    df_trva = dfx.head(trva)
    n = len(df_trva)
    tr = int(.9 * n)
    va = n - tr

    df_va = df_trva.head(va)
    df_tr = df_trva.tail(tr)

    df_xtest = df_test[['Home Team', 'Away Team', 'Date']]
    df_xtrain = df_tr[['Home Team', 'Away Team', 'Date']]
    df_xval = df_va[['Home Team', 'Away Team', 'Date']]

    df_ytest = df_test['Win']
    df_ytrain = df_tr['Win']
    df_yval = df_va['Win']

    df_xtest.to_csv(os.path.join(data_dir, 'x_testc.csv'))
    df_xval.to_csv(os.path.join(data_dir, 'x_valc.csv'))
    df_xtrain.to_csv(os.path.join(data_dir, 'x_trainc.csv'))

    df_ytest.to_csv(os.path.join(data_dir, 'y_testc.csv'))
    df_yval.to_csv(os.path.join(data_dir, 'y_valc.csv'))
    df_ytrain.to_csv(os.path.join(data_dir, 'y_trainc.csv'))

if __name__ == '__main__':
    train_test_split()