import pandas as pd
import os

root = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(root, 'cleaned_data')

csvs = os.listdir(data_dir)

#def train_test_split():
   