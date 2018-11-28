import sys

import pandas as pd


def main():
    pd.HDFStore('data/features.h5').remove(sys.argv[1])


if __name__ == '__main__':
    main()
