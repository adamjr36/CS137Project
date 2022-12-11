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
csvs = os.listdir(data_dir)

for csv in csvs:
    df = pd.read_csv(os.path.join(data_dir, csv))

    if df.empty:
        print(csv)
        assert(0)
    
    