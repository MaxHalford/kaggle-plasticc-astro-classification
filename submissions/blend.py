import pandas as pd


files = {
    '0.170_0.042_1.143_0.047.csv.gz': 1,
    '0.187_0.053_1.228_0.039.csv.gz': 1
}

subs = {file: pd.read_csv(file).set_index('object_id') for file in files}

blend = subs[list(subs.keys())[0]].copy()
blend[:] = 0

for sub in subs.values():
    blend += sub
blend /= len(subs)

blend.to_csv('blend.csv.gz', compression='gzip')
