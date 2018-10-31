import itertools

import pandas as pd
import numpy as np
from scipy import signal
from scipy import stats
from tsfresh.feature_extraction import feature_calculators as ts


# TODO: http://cesium-ml.org/docs/feature_table.html
# TODO: https://celerite.readthedocs.io/en/stable/


def symmetry(x):
    if len(x) < 2:
        return 0
    k = len(x) // 2
    left = x[:k]
    right = x[-k:]
    return np.abs(left - right).mean()


def number_of_peaks(x):
    x = x.rolling(window=5, center=True).mean()
    peaks = (x > x.mean() + x.std()).astype(int)
    n_switches = (peaks != peaks.shift()).sum() - 1
    return n_switches / 2


def basic_stats(x):
    x = np.sort(x)
    if len(x) < 3:
        return {}
    return {
        'mean': x.mean(),
        'min_1': x[0],
        'min_2': x[1],
        'min_2': x[2],
        'max_1': x[-1],
        'max_2': x[-2],
        'max_2': x[-3],
        'sem': stats.sem(x),
        'std': x.std(),
        'skew': stats.skew(x, bias=False),
        'kurtosis': stats.kurtosis(x, bias=False)
    }


def advanced_stats(x):
    return {
        'count_above_0': (x > 0).sum(),
        'count_below_0': (x < 0).sum(),
        'symmetry': symmetry(x),
        'number_of_peaks': number_of_peaks(x)
    }


def counts(x):
    return {
        'count': len(x),
        'count_above_mean': ts.count_above_mean(x),
        'count_below_mean': ts.count_below_mean(x)
    }


def diff_stats(x):
    d = x.diff()[1:]
    if len(d) > 0:
        return {f'diff_{k}': v for k, v in basic_stats(d).items()}
    return {}


def tests(x):
    if len(x) < 3:
        return {}
    return {
        'cid': ts.cid_ce(x, normalize=True),
        'tra': ts.time_reversal_asymmetry_statistic(x, lag=1),
        'shapiro_w': stats.shapiro(x)[0],
        'sample_entropy': ts.sample_entropy(x)
    }


def fourier(x):
    rfft = np.fft.rfft(x)
    amplitudes = np.abs(rfft)
    return {f'fourier_{k}': v for k, v in basic_stats(amplitudes).items()}


def lin_reg(x):
    lr = ts.linear_trend(x, param=[{'attr': 'slope'}, {'attr': 'intercept'}])
    return {
        'lr_slope': lr[0][1],
        'lr_intercept': lr[1][1]
    }


def detected_stats(x):
    return {
        'sum': x.sum(),
        'mean': x.mean()
    }


def mjd_stats(x):
    minimum = x.min()
    maximum = x.max()
    return {
        'min': minimum,
        'max': maximum,
        'ptp': maximum - minimum,
        'mean': x.mean(),
        'std': x.std(),
        'log_sum': np.log(x.sum())
    }


def percentiles(x):
    q = [5, 10, 25, 50, 75, 90, 95]
    p = np.percentile(x, q)
    return {f'p{qi}': v for qi, v in zip(q, p)}


def lab(x):
    """Should go in advances stats"""
    peaks, _ = signal.find_peaks(x)
    n = len(peaks)
    return {
        'n_peaks': n,
        'mean_peak_width': signal.peak_widths(x, peaks)[0].mean() if n > 0 else 0,
        'period_std': signal.periodogram(x)[0].std()
    }


AGGS = [
    # Solid gold
    (detected_stats, 'detected', ['object_id']),
    (advanced_stats, 'flux', ['object_id']),
    (basic_stats, 'flux', ['object_id']),
    (basic_stats, 'flux', ['object_id', 'detected']),
    (basic_stats, 'flux', ['object_id', 'passband']),
    (counts, 'flux', ['object_id', 'passband']),
    (diff_stats, 'flux', ['object_id', 'passband']),
    (basic_stats, 'flux_err', ['object_id']),
    (fourier, 'mjd', ['object_id']),
    (lin_reg, 'flux', ['object_id']),
    (mjd_stats, 'mjd', ['object_id', 'detected']),

    # Has to be fused
    (lab, 'flux', ['object_id', 'detected']),

    # To test
    (fourier, 'flux', ['object_id']),
    (diff_stats, 'mjd', ['object_id']),
]

def stream_groupby_csv(path, key, agg, chunk_size=1e6):

    # Make sure path is a list
    if not isinstance(path, list):
        path = [path]

    # Chain the chunks
    chunks = itertools.chain(*[pd.read_csv(p, chunksize=chunk_size) for p in path])

    results = []
    orphans = pd.DataFrame()

    for chunk in chunks:

        # Add the previous orphans to the chunk
        chunk = orphans.append(chunk)

        # Determine which rows are orphans
        last_val = chunk[key].iloc[-1]
        is_orphan = chunk[key] == last_val

        # Put the new orphans aside
        chunk, orphans = chunk[~is_orphan], chunk[is_orphan]

        results.append(agg(chunk))

    return pd.concat(results)


def main():

    # EAFP: Easier to Ask for Forgiveness than Permission
    store = pd.HDFStore('data/features.h5')

    # Load the light curves
    paths = [
        'data/training_set.csv',
        'data/test_set.csv'
    ]

    for (func, on, by) in AGGS:

        name = func.__name__
        key = f'{on}_{name}_by_{"_and_".join(by)}'

        # Don't compute the features if they're already saved
        if key in store:
            print(f'Skipping {key}')
            continue

        print(f'Making {key}...')

        # Compute aggregates
        agg = lambda g: g.groupby(by)[on].apply(func)
        features = stream_groupby_csv(path=paths, key='object_id', agg=agg, chunk_size=2e6)

        # Unstack until the features are indexed by object_id
        while isinstance(features.index, pd.MultiIndex):
            features = features.unstack()

        # Collapse the column names
        if isinstance(features.columns, pd.MultiIndex):
            features.columns = [
                '_'.join([
                    f'{features.columns.names[i] or ""}_{v}'
                    for i, v in enumerate(col)
                ])
                for col in features.columns.values
            ]

        # Add a prefix to know by which variables the aggregation occurred
        features = features.add_prefix(f'{on}_')
        features.columns = [c.replace('__', '_') for c in features.columns]

        # Save the features
        store.put(key, features)

        # Plays a beep sound
        print('\a')


if __name__ == '__main__':
    main()
