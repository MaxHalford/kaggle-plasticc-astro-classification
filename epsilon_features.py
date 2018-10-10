"""
References:

- http://docs.astropy.org/en/stable/api/astropy.stats.LombScargle.html
- https://jakevdp.github.io/blog/2015/06/13/lomb-scargle-in-python/
- https://jakevdp.github.io/blog/2017/03/30/practical-lomb-scargle/
"""

import csv
import collections
import contextlib
import itertools

import astropy.stats as stats
import numpy as np
from scipy.stats import shapiro
import tqdm
import upsilon


def calc_period(time, flux, flux_err):
    if len(flux) < 3:
        return -1
    frequency, power = stats.LombScargle(time, flux, flux_err).autopower(nyquist_factor=20)
    return frequency[power.argmax()]


def calc_shapiro_w(flux):
    if len(flux) < 3:
        return -1
    w, _ = shapiro(flux)
    return w


def calc_q1_q3_delta(flux):
    if len(flux) < 3:
        return -1
    q1, q3 = np.percentile(flux, [25, 75])
    return q3 - q1


FEATURES = {
    'period': lambda t, y, dy: calc_period(t, y, dy),
    'shapiro_w': lambda t, y, dy: calc_shapiro_w(y),
    'q1_q3_delta': lambda t, y, dy: calc_q1_q3_delta(y),
}


def get_features(t, y, dy):
    t = np.array(t)
    y = np.array(y)
    dy = np.array(dy)
    try:
        e_features = upsilon.ExtractFeatures(t, y, dy, n_threads=8)
        e_features.run()
        return e_features.get_features()
    except (TypeError, ValueError):
        return {}


def main():

    with contextlib.ExitStack() as stack:
        train = stack.enter_context(open('data/training_set.csv', 'r'))
        test = stack.enter_context(open('data/test_set.csv', 'r'))
        file = stack.enter_context(open('data/features/object_passband.csv', 'w'))

        # The data is stored in multiple dictionaries where each key is a passband
        times = collections.defaultdict(list)
        fluxes = collections.defaultdict(list)
        errors = collections.defaultdict(list)

        object_id = None

        writer = None

        for row in tqdm.tqdm(itertools.chain(csv.DictReader(train), csv.DictReader(test))):

            # Check to see if we've collected all the data pertaining to the current object
            if object_id and object_id != row['object_id']:

                # Compute and save features
                for passband in fluxes:

                    t = np.array(times[passband])
                    y = np.array(fluxes[passband])
                    dy = np.array(errors[passband])
                    features = get_features(t, y, dy)

                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=['object_id', 'passband'] + list(features.keys()))
                        writer.writeheader()

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

        # Let's not forget the last (object_id, fluxes) pair
        for passband in fluxes:
            t = np.array(times[passband])
            y = np.array(fluxes[passband])
            dy = np.array(errors[passband])
            features = get_features(t, y, dy)

            writer.writerow(dict(
                **{'object_id': object_id, 'passband': passband},
                **features
            ))


if __name__ == '__main__':
    main()
