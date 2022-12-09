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
    def convert(x):
        x = x.split(':')
        hw = x[0]
        aw = x[1]
        if hw > aw:
            return 1
        elif aw > hw:
            return -1
        return 0
    df['Home Win'] = df['Score'].map(convert)
    

    df = df.drop(columns=['Home Team', 'Away Team', 'Unnamed: 0', 'Score', 'Match', 'Competition', 'Duration', 'Team'])

    df.to_csv(os.path.join(data_dir, csv))


