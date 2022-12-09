import pandas as pd
import os

root = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(root, 'cleaned_data')

csvs = os.listdir(data_dir)

def fix_data():
    for csv in csvs:
        df = pd.read_csv(os.path.join(data_dir, csv))

        if df.empty:
            print(csv)
            assert(0)

        #Change 'home' and 'away' and 'home' with 1 or 0
        team = csv.split('_')[0]
        df['Home'] = 1 if (df['Home Team'] == team) else 0
        df = df.drop(columns=['Home Team', 'Away Team'])

#def train_test_split():
   