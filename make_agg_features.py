import collections
import itertools

from cesium import featurize
from cesium import time_series
import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
from tsfresh.feature_extraction import feature_calculators as ts


Agg = collections.namedtuple('Aggregate', 'on by how name')
Agg.__new__.__defaults__ = (None,) * len(Agg._fields)

PASSBAND_MAPPING = {
    '0': 'u',
    '1': 'g',
    '2': 'r',
    '3': 'i',
    '4': 'z',
    '5': 'y',
}


def lin_reg(x):
    lr = ts.linear_trend(x, param=[{'attr': 'slope'}, {'attr': 'intercept'}])
    return {
        'lr_slope': lr[0][1],
        'lr_intercept': lr[1][1]
    }


def rcs(x):
    """From Ellaway 1978"""
    sigma = np.std(x)
    N = len(x)
    m = np.mean(x)
    s = np.cumsum(x - m) * 1.0 / (N * sigma)
    R = np.max(s) - np.min(s)
    return R


def stetson_k(x):
    magnitude = x['flux']
    error = x['flux_err']

    mean_mag = (np.sum(magnitude/(error*error)) /
                np.sum(1.0 / (error * error)))

    N = len(magnitude)
    sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
              (magnitude - mean_mag) / error)

    K = (1 / np.sqrt(N * 1.0) *
         np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))

    return K


def max_diff(x):
    i = x['flux'].idxmax()
    a = x['mjd'].iloc[0]
    b = x['mjd'].loc[i]
    return b - a


# def normalize(x):
#     return (x - x.mean()) / x.std()

# def periodic_features(df):

#     bands = df.groupby('passband')

#     ts = time_series.TimeSeries(
#         t=bands['mjd'].apply(lambda x: x.values),
#         m=bands['flux'].apply(lambda x: normalize(x.values)),
#         e=bands['flux_err'].apply(lambda x: x.values)
#     )

#     periods = featurize.featurize_single_ts(ts, features_to_use=['freq1_freq'])['freq1_freq'].to_dict()
#     features = {}

#     for i, (mjd, flux, flux_err) in enumerate(ts.channels()):

#         mjd = (periods[i] * mjd) % 1
#         n = flux.size
#         m = flux.argmin()

#         flux = np.roll(flux, n - m)
#         mjd = np.roll(mjd, n - m)
#         mjd = (mjd + (1 - mjd[0])) % 1

#         features[f'variation_coeff_{i}'] = np.std(np.diff(flux)) / abs(np.mean(np.diff(flux)))
#         features[f'skew_{i}'] = stats.skew(flux)
#         features[f'autocorr_{i}'] = pd.Series(flux).autocorr()
#         features[f'ptp_{i}'] = np.ptp(flux)
#         features[f'n_peaks_{i}'] = len(signal.find_peaks(flux, prominence=flux.mean())[0])

#     return pd.Series(features)


def avg_double_to_single_step(df):

    # http://cesium-ml.org/docs/feature_table.html

    bands = df.groupby('passband')

    ts = time_series.TimeSeries(
        t=bands['mjd'].apply(lambda x: x.values),
        m=bands['flux'].apply(lambda x: x.values),
        e=bands['flux_err'].apply(lambda x: x.values)
    )

    name = 'avg_double_to_single_step'
    features = featurize.featurize_single_ts(ts, features_to_use=[name])[name].to_dict()
    return pd.Series(features)


# My best model has hard time separating classes 42, 52, and to some extent, 67 and 90.
# Separating these classes is the key to winning the competition, more than class 99 IMHO.

from scipy import stats

def detected_lr(g):
    d = g.query('detected == True')
    slope, *_ = stats.linregress(d['mjd'], d['flux'])
    return slope


AGGS = [
    Agg(on=['flux', 'mjd', 'detected'], by='object_id', how=detected_lr),

    Agg(on='flux', by='object_id', how='mean'),
    Agg(on='flux', by=['object_id', 'passband'], how='mean'),
    Agg(on='flux', by=['object_id', 'passband'], how='std'),
    Agg(on='mjd', by=['object_id', 'detected'], how=np.ptp),
    Agg(on='mjd', by=['object_id', 'detected'], how='std'),
    Agg(on='mjd', by=['object_id', 'passband', 'detected'], how=np.ptp),
    Agg(on='flux', by='object_id', how='median'),
    Agg(on='detected', by='object_id', how='mean'),
    Agg(on='flux', by='object_id', how='skew'),
    Agg(on='flux', by=['object_id', 'passband'], how='skew'),
    Agg(on='flux', by=['object_id', 'detected'], how='min'),
    Agg(on='flux', by=['object_id', 'passband'], how='max'),
    Agg(on='flux', by=['object_id', 'detected'], how=np.ptp),
    Agg(on='mjd', by=['object_id', 'detected'], how=lambda x: x.diff().mean(), name='diff'),
    Agg(on='flux', by=['object_id', 'detected'], how=lin_reg),
    Agg(on='flux', by='object_id', how=rcs),
    Agg(on=['flux', 'flux_err'], by='object_id', how=stetson_k),
    Agg(on='mjd', by=['object_id', 'detected'], how=lambda x: np.log(np.sum(x)), name='log_sum'),
    Agg(on='flux', by=['object_id', 'detected'], how=lambda x: {'below_0': (x < 0).sum(), 'above_0': (x > 0).sum()}, name='level'),
    Agg(on='flux', by=['object_id', 'detected'], how=lambda x: stats.skew(np.cumsum(x)), name='cumsum_skew'),
    Agg(on='flux', by=['object_id', 'detected'], how=lambda x: np.std(np.abs(x)), name='abs_std'),
    Agg(on=['flux', 'mjd'], by=['object_id', 'detected'], how=max_diff),
    Agg(on='flux_err', by=['object_id', 'detected'], how='min'),
]


def stream_groupby_csv(paths, key, gb, chunk_size=1e6, **kwargs):

    # Chain the chunks
    kwargs['chunksize'] = chunk_size
    chunks = itertools.chain(*[pd.read_csv(p, **kwargs) for p in paths])

    results = []
    orphans = pd.DataFrame()

    for chunk in chunks:

        # Add the previous orphans to the chunk
        chunk = pd.concat((orphans, chunk))

        # Determine which rows are orphans
        last_val = chunk[key].iloc[-1]
        is_orphan = chunk[key] == last_val

        # Put the new orphans aside
        chunk, orphans = chunk[~is_orphan], chunk[is_orphan]

        results.append(gb(chunk))

    return pd.concat(results)


def main():

    store = pd.HDFStore('data/features.h5')

    paths = [
        '~/projects/kaggle-plasticc-astro-classification/data/kaggle/training_set.csv',
        #'~/projects/kaggle-plasticc-astro-classification/data/kaggle/test_set.csv'
    ]

    keys = set()

    for agg in AGGS:

        on = agg.on
        by = agg.by
        how = agg.how
        name = agg.name or (how if isinstance(how, str) else how.__name__)

        # Build the key used to store the features
        prefix = '_'.join(on) if isinstance(on, list) else on
        suffix = f'{name}_by_{"_and_".join(by) if isinstance(by, list) else by}'
        key = f'{prefix}_{suffix}'
        keys.add(key)

        # Don't compute the features if already done
        if key in store:
            print(f'Skipping {key}')
            continue
        print(f'Computing {key}...')

        # Compute the features
        if isinstance(how, str):
            gb = lambda chunk: chunk.groupby(by)[on].agg(how)
        else:
            gb = lambda chunk: chunk.groupby(by)[on].apply(how)
        features = stream_groupby_csv(
            paths=paths,
            key='object_id',
            gb=gb,
            converters={
                'passband': lambda x: PASSBAND_MAPPING[x],
                #'mjd': lambda x: int(x.split('.')[0]),
            }
        )

        # Make sure features is a DataFrame
        if isinstance(features, pd.Series):
            features = features.to_frame('')

        # Unstack until the features have a single index
        while isinstance(features.index, pd.MultiIndex):
            features = features.unstack()

        # Collapse the column names
        if isinstance(features.columns, pd.MultiIndex):
            features.columns = [
                '_'.join(str(c) for c in col if c != '')
                for col in features.columns.values
            ]

        # Use appropriate names to identify the features
        features = features.add_prefix(f'{prefix}_')
        features = features.add_suffix(f'_{suffix}')
        features.columns = [c.replace('__', '_') for c in features.columns]

        # Save the features
        store.put(key, features)


    # Remove the features in the store that are not in the list of aggregations
    for key in set(k[1:] for k in store.keys()).difference(keys):
        #if key in ['encoded']:
        #    continue
        print(f'Removing {key}')
        store.remove(key)


if __name__ == '__main__':
    main()
