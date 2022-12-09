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
    
    team = df['Team'].values[0]
    df['Home'] = df['Home Team'].map(lambda x: 1 if (x == team) else 0)

    #Drop % from scheme
    df['Scheme'] = df['Scheme'].str.split().str[0]
    
    #convert score
    def convert(row):
        score = row['Score'].split(':')
        hw = score[0]
        aw = score[1]
        if hw == aw:
            return 0
        if hw > aw:
            if row['Home'] == 1:
                return 1
            
        else:
            if row['Home'] == 0:
                return 1
        return -1

    df['Win'] = df.apply(lambda row: convert(row), axis=1)
    

    df = df.drop(columns=['Unnamed: 0', 'Score', 'Competition', 'Duration'])
    df.to_csv(os.path.join(data_dir, csv))


