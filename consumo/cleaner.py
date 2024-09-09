import os.path
import sys

import pandas as pd

file = sys.argv[1]

df = pd.read_csv(os.path.join(os.getcwd(), file))
df.set_index('TIME', inplace=True)

df.drop(columns=['CH1', 'CH2', 'TIME.1', 'Unnamed: 4'], inplace=True)

df = df[df.index < 4.7]

df.to_csv('32_data_clean.csv')
