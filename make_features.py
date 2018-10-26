import itertools

import pandas as pd
import numpy as np
from scipy import stats
from tsfresh.feature_extraction import feature_calculators as ts


# TODO: http://cesium-ml.org/docs/feature_table.html
# TODO: https://celerite.readthedocs.io/en/stable/


def lin_reg(x):
    lr = ts.linear_trend(x, param=[{'attr': 'slope'}, {'attr': 'intercept'}])
    return {
        'flux_slope': lr[0][1],
        'flux_intercept': lr[1][1]
    }


def percentiles(x):
    q = [5, 10, 25, 50, 75, 90, 95]
    p = np.percentile(x, q)
    return {f'flux_p{qi}': v for qi, v in zip(q, p)}


def fourier(x):
    fft = np.fft.fft(x)
    amplitudes = np.sort(np.abs(fft))
    return {
        'flux_fft_amp_0': amplitudes[-1],
        'flux_fft_amp_1': amplitudes[-2] if len(amplitudes) > 1 else 0,
        'flux_fft_amp_2': amplitudes[-3] if len(amplitudes) > 2 else 0,
        'flux_fft_mean': amplitudes.mean(),
        'flux_fft_std': amplitudes.std()
    }


def diff_stats(x):
    d = x.diff()
    return {
        'flux_diff_mean', d.mean(),
        'flux_diff_min', d.min(),
        'flux_diff_max', d.max(),
        'flux_diff_std', d.std(),
        'flux_diff_pos_count', (d > 0).sum(),
        'flux_diff_neg_count', (d < 0).sum()
    }


AGGS = {
    'flux_counts': lambda df: df.groupby(['object_id', 'passband'])['flux'].agg([
        ('flux_count', len),
        ('flux_count_above_mean', ts.count_above_mean),
        ('flux_count_below_mean', ts.count_below_mean),
    ]),
    'flux_diff_stats': lambda df: df.groupby(['object_id', 'passband'])['flux']\
                                    .apply(diff_stats)\
                                    .unstack(),
    'flux_err_stats': lambda df: df.groupby(['object_id', 'passband'])['flux_err'].agg([
        ('flux_err_mean', 'mean'),
        ('flux_err_std', 'std'),
        ('flux_err_shapiro_w', lambda x: stats.shapiro(x)[0] if len(x) > 2 else 0)
    ]),
    'detected_stats': lambda df: df.groupby(['object_id', 'passband'])['detected'].agg([
        ('detected_mean', 'mean'),
    ]),
    'flux_fourier': lambda df: df.groupby(['object_id', 'passband'])['flux']\
                                 .apply(fourier)\
                                 .unstack(),
    'flux_lin_reg': lambda df: df.groupby(['object_id', 'passband'])['flux']\
                                 .apply(lin_reg)\
                                 .unstack(),
    'flux_percentiles': lambda df: df.groupby(['object_id', 'passband'])['flux']\
                                     .apply(percentiles)\
                                     .unstack(),
    'flux_sample_entropy': lambda df: df.groupby(['object_id', 'passband'])['flux'].agg([
        ('flux_sample_entropy', ts.sample_entropy),
    ]),
    'flux_stats': lambda df: df.groupby(['object_id', 'passband'])['flux'].agg([
        ('flux_mean', 'mean'),
        ('flux_min', 'min'),
        ('flux_max', 'max'),
        ('flux_std', 'std'),
        ('flux_skew', lambda x: stats.skew(x, bias=False)),
        ('flux_kurtosis', lambda x: stats.kurtosis(x, bias=False))
    ]),
    'flux_tests': lambda df: df.groupby(['object_id', 'passband'])['flux'].agg([
        ('flux_cid', lambda x: ts.cid_ce(x, normalize=True)),
        ('flux_tra', lambda x: ts.time_reversal_asymmetry_statistic(x, lag=1)),
        ('flux_shapiro_w', lambda x: lambda x: stats.shapiro(x)[0])
    ]),
    'object_stats': lambda df: df.groupby('object_id').agg({
        'flux': [
            ('mean', 'mean'),
            ('median', 'median'),
            ('min', 'min'),
            ('max', 'max'),
            ('std', 'std'),
            ('skew', lambda x: stats.skew(x, bias=False)),
            ('kurtosis', lambda x: stats.kurtosis(x, bias=False))
        ],
        'detected': [
            ('mean', 'mean')
        ]
    }),
    'object_detected': lambda df: df.groupby(['object_id', 'detected'])['mjd'].agg([
        ('mjd_ptp', np.ptp),
        ('mjd_min', np.min),
        ('mjd_max', np.max)
    ])
}

def stream_groupby_csv(path, key, agg, chunk_size=1e6):

    # Make sure path is a list
    if not isinstance(path, list):
        path = [path]

    results = pd.DataFrame()
    orphans = pd.DataFrame()

    # Chain the chunks
    chunks = itertools.chain(*[pd.read_csv(p, chunksize=chunk_size) for p in path])

    for chunk in chunks:

        # Determine which rows are orphans
        last_val = chunk[key].iloc[-1]
        is_orphan = chunk[key] == last_val

        # Put the new orphans aside and add the previous orphans to the chunk
        chunk, orphans = chunk[~is_orphan].append(orphans), chunk[is_orphan]

        results = results.append(agg(chunk))

    return results


def main():

    # EAFP: Easier to Ask for Forgiveness than Permission
    store = pd.HDFStore('data/features.h5')

    # Load the light curves
    paths = [
        'data/training_set.csv',
        'data/test_set.csv'
    ]

    for name, agg in AGGS.items():

        # Don't compute the features if they're already saved
        if name in store:
            print(f'Skipping {name}')
            continue

        # Compute the features
        print(f'Making {name}...')
        features = stream_groupby_csv(path=paths, key='object_id', agg=agg, chunk_size=1e6)

        if isinstance(features.index, pd.MultiIndex):
            features = features.unstack()

        names = features.columns.get_level_values(0)
        passbands = features.columns.get_level_values(1)
        features.columns = [f'{n}_{p}' for (n, p) in zip(names, passbands)]

        # Save the features
        store.put(name, features)


if __name__ == '__main__':
    main()
