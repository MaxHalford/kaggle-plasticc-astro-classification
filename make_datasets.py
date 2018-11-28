import itertools

import numpy as np
import pandas as pd


def main():

    # Load the metadata
    df = pd.concat(
        (
            pd.read_csv('~/projects/kaggle-plasticc-astro-classification/data/kaggle/training_set_metadata.csv'),
            pd.read_csv('~/projects/kaggle-plasticc-astro-classification/data/kaggle/test_set_metadata.csv')
        ),
        ignore_index=True,
        sort=False
    )
    df['is_train'] = df['target'].notnull()
    df['is_galactic'] = df['hostgal_photoz'] == 0

    # Photometric redshift
    df['hostgal_photoz_min'] = df.eval('hostgal_photoz - hostgal_photoz_err')
    df['hostgal_photoz_max'] = df.eval('hostgal_photoz + hostgal_photoz_err')
    df['hostgal_photoz_over_err'] = df.eval('hostgal_photoz / (hostgal_photoz_err + 1)')

    # Load the aggregate features
    with pd.HDFStore('data/features.h5') as store:
        for key in store:
            print(key)
            df = df.join(store.get(key).astype(np.float32), on='object_id')

    for (i, j) in itertools.combinations(['u', 'g', 'r', 'i', 'z', 'y'], 2):

        df[f'mjd_1_{i}_{j}_ptp_by_object_id_and_passband_and_detected'] = df.eval(f'mjd_1_{i}_ptp_by_object_id_and_passband_and_detected / (1 + mjd_1_{j}_ptp_by_object_id_and_passband_and_detected)')

        for stat in ['mean', 'skew', 'max', 'std']:
            df[f'flux_{i}_{j}_{stat}_by_object_id_and_passband'] = df.eval(f'flux_{i}_{stat}_by_object_id_and_passband / (1 + flux_{j}_{stat}_by_object_id_and_passband)')

    # Make datasets
    to_drop = ['is_train', 'hostgal_specz']

    train = df[df['is_train']]
    train = train.drop(columns=to_drop).reset_index(drop=True)
    train['target'] = train['target'].apply(lambda x: f'class_{int(x)}').astype('category')

    test = df[~df['is_train']].reset_index(drop=True)
    test = test.drop(columns=to_drop + ['target'])

    # Save datasets
    train.to_feather('data/train.fth')
    test.to_feather('data/test.fth')


if __name__ == '__main__':
    main()
