import pandas as pd
import os
from columns import column_name_changes

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


data_folder = os.path.split(os.path.split(os.getcwd())[0])[0]
df = pd.read_excel(os.path.join(data_folder, 'Data/Team Stats Southampton 18-23.xlsx'))
df.rename(columns=column_name_changes, inplace=True)
print(df.iloc[1])

'''
for file in filenames.txt:...
'''