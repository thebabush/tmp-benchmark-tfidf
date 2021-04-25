#!/usr/bin/env python3

import json
import sys

import pandas as pd


COLUMNS = ['density', 'm', 'n', 'mean', 'std', 'version']


def main(old, new, merged):
    old = pd.read_json(old)
    new = pd.read_json(new)
    old['version'] = 'old'
    new['version'] = 'new'

    df = pd.concat([old, new])

    newold = new[COLUMNS].merge(
        old[COLUMNS],
        on=['density', 'm', 'n'],
        suffixes=('_new', '_old')
    )
    newold.to_csv(merged, index=False)


if __name__ == '__main__':
    main(*sys.argv[1:])
