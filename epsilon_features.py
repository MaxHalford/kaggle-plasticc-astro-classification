"""
References:

- http://docs.astropy.org/en/stable/api/astropy.stats.LombScargle.html
- https://jakevdp.github.io/blog/2015/06/13/lomb-scargle-in-python/
- https://jakevdp.github.io/blog/2017/03/30/practical-lomb-scargle/
"""

import csv
import collections
from concurrent import futures
import contextlib
import copy
import itertools

import astropy.stats as stats
import numpy as np
from scipy.stats import shapiro
import tqdm
import upsilon


def make_features(t, y, dy):
    try:
        e_features = upsilon.ExtractFeatures(t, y, dy, n_threads=8)
        e_features.run()
        return e_features.get_features()
    except (TypeError, ValueError):
        return None


def main():

    with contextlib.ExitStack() as stack:
        train = stack.enter_context(open('data/training_set.csv', 'r'))
        test = stack.enter_context(open('data/test_set.csv', 'r'))
        file = stack.enter_context(open('data/features/epsilon.csv', 'w'))

        # The data is stored in multiple dictionaries where each key is a passband
        times = collections.defaultdict(list)
        fluxes = collections.defaultdict(list)
        errors = collections.defaultdict(list)

        object_id = None
        writer = csv.DictWriter(
            file,
            fieldnames=['object_id', 'passband', 'amplitude', 'cusum', 'eta', 'hl_amp_ratio',
                        'kurtosis', 'n_points', 'period', 'period_SNR', 'period_log10FAP',
                        'period_uncertainty', 'phase_cusum', 'phase_eta', 'phi21', 'phi31',
                        'quartile31', 'r21', 'r31', 'shapiro_w', 'skewness', 'slope_per10',
                        'slope_per90', 'stetson_k', 'weighted_mean', 'weighted_std']
        )
        writer.writeheader()

        for row in tqdm.tqdm(itertools.chain(csv.DictReader(train), csv.DictReader(test))):

            # Check to see if we've collected all the data pertaining to the current object
            if object_id and object_id != row['object_id']:

                for passband in fluxes:
                    t = times[passband]
                    y = fluxes[passband]
                    dy = errors[passband]
                    features = make_features(t, y, dy)
                    if features:
                        writer.writerow(dict(
                            **{'object_id': object_id, 'passband': passband},
                            **features
                        ))

                # Reset the data and move to the next object
                times = collections.defaultdict(list)
                fluxes = collections.defaultdict(list)
                errors = collections.defaultdict(list)

            # Update the current object_id (even if this doesn't change)
            object_id = row['object_id']

            # Parse the current row's fields
            times[row['passband']].append(float(row['mjd']))
            fluxes[row['passband']].append(float(row['flux']))
            errors[row['passband']].append(float(row['flux_err']))

        # Let's not forget the last object
        for passband in fluxes:
            t = times[passband]
            y = fluxes[passband]
            dy = errors[passband]
            features = make_features(t, y, dy)
            if features:
                writer.writerow(dict(
                    **{'object_id': object_id, 'passband': passband},
                    **features
                ))


if __name__ == '__main__':
    main()
