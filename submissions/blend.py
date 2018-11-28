import pandas as pd


blend = None

files = {
    '0.058_0.044_0.791_0.028_scirpus.csv.gz': 1,
    'single_subm_0.654771_2018-11-20-04-44.csv': 1
}

for file, weight in files.items():
    sub = pd.read_csv(file).set_index('object_id')
    if blend is None:
        blend = weight * sub
    else:
        blend += weight * sub

blend /= sum(files.values())

def GenUnknown(data):
    return ((((((data["mymedian"]) + (((data["mymean"]) / 2.0)))/2.0)) + (((((1.0) - (((data["mymax"]) * (((data["mymax"]) * (data["mymax"]))))))) / 2.0)))/2.0)

feats = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
         'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
         'class_92', 'class_95']

y = pd.DataFrame()
y['mymean'] = blend[feats].mean(axis=1)
y['mymedian'] = blend[feats].median(axis=1)
y['mymax'] = blend[feats].max(axis=1)

blend['class_99'] = GenUnknown(y)

blend.to_csv('blend.csv.gz', compression='gzip')
