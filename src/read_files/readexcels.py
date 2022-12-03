import pandas as pd
import os
from columns import column_name_changes

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

teams = {}
data = []

filenames = []
with open('filenames.txt') as file:
    for line in file:
        filenames.append(line.strip())


data_folder = os.path.split(os.path.split(os.getcwd())[0])[0]

filename = 'Team Stats Southampton 18-23.xlsx'

for filename in filenames:
    df = pd.read_excel(os.path.join(data_folder, 'Data/' + filename))
    df.rename(columns=column_name_changes, inplace=True)

    team = ''.join(filename.split()[2:-1])

    #row = df.iloc[1]
    #match = row['Match'].split('-')
    #home = match[0].strip()
    #away = ''.join((match[1].split()[:-1])).strip()


    data.append(df)
    teams[team] = df[df['Team'] == team]

data = pd.concat(data)

data.to_csv('master_data.csv')
for key in teams:
    filename = key + '_data.csv'
    teams[key].to_csv(filename)