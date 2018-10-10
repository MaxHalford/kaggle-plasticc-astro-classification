import pandas as pd


files = {
    'naive_benchmark.csv.gz': 1,
    'naive_benchmark.csv': 1
}

subs = {file: pd.read_csv(file, index_col=0) for file in files}

blend = subs[list(subs.keys())[0]].copy()
blend[:] = 0

for sub in subs.values():
    print(sub.head())
    blend += sub
blend /= len(subs)

print(blend.head())

blend.to_csv('blend.csv.gz', compression='gzip')
