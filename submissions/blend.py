import pandas as pd


files = {
    'lol_0.095_0.034_1.041_0.032.csv.gz': 1,
    'lol_0.131_0.037_1.072_0.040.csv.gz': 1
}

subs = {file: pd.read_csv(file).set_index('object_id') for file in files}

blend = subs[list(subs.keys())[0]].copy()
blend[:] = 0

for sub in subs.values():
    blend += sub
blend /= len(subs)

blend.to_csv('blend.csv.gz', compression='gzip')
