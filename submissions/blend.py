import pandas as pd


files = {
    '0.073_0.025_0.852_0.043.csv.gz': 1,
    '0.072_0.028_0.839_0.035.csv.gz': 1
}

subs = {file: pd.read_csv(file).set_index('object_id') for file in files}

blend = subs[list(subs.keys())[0]].copy()
blend[:] = 0

for sub in subs.values():
    blend += sub
blend /= len(subs)

print(blend.head())

blend.to_csv('blend.csv.gz', compression='gzip')
