import numpy as np
import pandas as pd


def main():

    dtypes = {
        'object_id': np.int32,
        'mjd': np.float32,
        'passband': 'category',
        'flux': np.float32,
        'flux_err': np.float32,
        'detected': bool
    }

    lcs = pd.concat(
        (
            pd.read_csv('data/training_set.csv', dtype=dtypes),
            pd.read_csv('data/test_set.csv', dtype=dtypes)
        ),
        sort=False,
        ignore_index=True
    )

    passband_map = {
        0: 'u',
        1: 'g',
        2: 'r',
        3: 'i',
        4: 'z',
        5: 'y'
    }

    lcs['passband'] = lcs['passband'].map(passband_map).astype('category')

    lcs.to_feather('data/light_curves.fth')

if __name__ == '__main__':
    main()
