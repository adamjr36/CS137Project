import pandas as pd
import os

root = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(root, 'cleaned_data')

csvs = os.listdir(data_dir)

for csv in csvs:
    df = pd.read_csv(os.path.join(data_dir, csv))

    if df.empty:
        print(csv)
        assert(0)

    match = df['Match'].astype('string')
    match = match.str.split('-')
    home = match.str[0].str.strip()
    away = match.str[1].str.split().str[:-1].str.join(' ').str.strip()
    score = match.str[1].str.split().str[-1].str.strip()
    df['Home Team'] = home
    df['Away Team'] = away
    df['Score'] = score
    df = df.drop(columns=['Unnamed: 0'])

    df.to_csv(os.path.join(data_dir, csv))


