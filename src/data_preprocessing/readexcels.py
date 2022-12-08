import pandas as pd
import os
from columns import column_name_changes

team_names = {"Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Burnley", 
"Cardiff City", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Huddersfield Town",
"Leeds United", "Leicester City", "Liverpool", "Manchester City", "Manchester United",
"Newcastle United", "Norwich City", "Nottingham Forest", "Sheffield United", "Southampton", 
"Tottenham Hotspur", "Watford", "West Bromwich Albion", "West Ham United", "Wolverhampton Wanderers"}

root = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(root, 'cleaned_data')

teams = {}
data = []

filenames = []
with open('filenames.txt') as file:
    for line in file:
        filenames.append(line.strip())


data_folder = os.path.split(os.path.split(os.getcwd())[0])[0]

for filename in filenames:
    df = pd.read_excel(os.path.join(data_folder, 'Data/' + filename))
    df.rename(columns=column_name_changes, inplace=True)

    team = ""
    for team_name in team_names:
        if team_name in filename:
            team = team_name
            break
    print(filename)
    assert(team != "")

    data.append(df)
    teams[team] = df[df['Team'].str.strip() == team.strip()]
    if teams[team].empty:
        print(team)
        print("Empty!")
        assert(0)

data = pd.concat(data)

data.to_csv(os.path.join(data_dir, 'master_data.csv'))
for key in teams:
    filename = key + '_data.csv'
    teams[key].to_csv(os.path.join(data_dir, filename))