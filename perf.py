#!/usr/bin/env python3

import itertools
import json
import numpy as np
import random
import scipy as sp
import sys
import time

from memory_profiler import memory_usage
from sklearn.feature_extraction.text import TfidfTransformer


DENSITY = np.linspace(0.01, 0.5, 5)[::-1]
M = np.logspace(np.log10(10), np.log10(10000), 5, dtype=np.uint64)[::-1]
N = np.logspace(np.log10(10), np.log10(10000), 5, dtype=np.uint64)[::-1]
MIN_ITER = 5
MAX_ITER = 100
PRECISION = 0.01


def mk(m, n, d):
    random.seed(666)
    X = sp.sparse.random(1000, n, density=d, format='csr')
    Y = sp.sparse.random(m, n, density=d, format='csr')
    tfidf = TfidfTransformer()
    tfidf.fit(X)

    def doit():
        t0 = time.perf_counter()
        tfidf.transform(Y, copy=True)
        return time.perf_counter() - t0

    return doit


def main(out_path):
    results = []
    for density, m, n in itertools.product(DENSITY, M, N):
        m = int(m)
        n = int(n)
        density = float(density)

        measures = []
        mems = []
        f = mk(m, n, density)
        for i in range(MAX_ITER):
            mem, diff = memory_usage(f, retval=True)
            mems.append(mem)
            measures.append(diff)
            mean = np.mean(measures)
            std = np.std(measures)
            if len(measures) > MIN_ITER and std / mean < PRECISION:
                break

        print(f'm={m:10},n={n:10},density={density:0.9f}  -  {np.mean(measures):0.9f} +/- {np.std(measures):0.9f}', std / mean)

        results.append({
            'density': density,
            'm': m,
            'n': n,
            'mean': mean,
            'std': std,
            'measures': measures,
            'memory': mems,
        })

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main(*sys.argv[1:])
