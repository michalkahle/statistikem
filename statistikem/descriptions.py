import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import statsmodels.api as sm
from scipy import stats
import warnings
import re
from math import ceil

from statistikem import helpers
from statistikem import comparisons
from statistikem.comparisons import ALPHA

def describe(data):
    results = []
    n_rows = ceil(data.shape[1] / 6)
    fig, ax = plt.subplots(n_rows, 6, figsize=(12, n_rows*2), dpi=75)
    ax = ax.flatten()
    for n, col in enumerate(data.columns):
        s = data[col]
        nona = s.dropna()
        scale = helpers.guess_scale(nona)
        res = {'var': col, 'scale': scale}
        if scale == 'binary':
            counts = pd.crosstab(s, np.ones(len(s)))
            comparisons._plot_bars(counts.T, ax[n])
            count = counts.iloc[:,0]
            res['description'] = f'{count.iloc[-1]}/{count.sum()} ({count.iloc[-1] / count.sum() * 100:2.0f}%)'
        elif scale == 'categorical' or scale == 'continuous':
            p, lp = comparisons.test_for_normality(s)
            possibly_normal = p > ALPHA
            possibly_lognormal = lp > ALPHA
            if (not possibly_normal) and possibly_lognormal:
                warnings.warn(f'Variable "{col}" might have lognormal distribution.')
            comparisons._plot_histograms(nona, [nona], [possibly_normal], [possibly_lognormal], [ax[n]])
            if possibly_normal:
                res['description'] = helpers.format_float(np.mean(s)) + ' ±' + helpers.format_float(np.std(s, ddof=1))
            else:
                p25, p50, p75 = np.percentile(s, [25, 50, 75], method='midpoint')
                res['description'] = f'{helpers.format_float(p50)} ({helpers.format_float(p25)}, {helpers.format_float(p75)})'

        ax[n].set_title(col)
        results.append(res)
    fig.tight_layout()
    return pd.DataFrame(results)
