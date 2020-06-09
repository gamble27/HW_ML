import numpy as np
import pandas as pd

df = pd.read_csv('http://www.occamslab.com/petricek/data/ratings.dat',
                   names=['userID', 'itemID', 'rating'])
df1 = df.iloc[50000:70000]
df1.to_csv('/home/olga/Projects/HW_ML/alice/brand_new_data.csv')
