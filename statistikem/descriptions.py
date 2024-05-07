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

def describe(data, parametric=None, scales=None, plot=True):
    results = []
    if type(data) == pd.core.series.Series:
        data = pd.DataFrame(data)
    if type(scales) != list:
        scales = [scales] * data.shape[1]
    if plot:
        n_rows = ceil(data.shape[1] / 6)
        fig, ax = plt.subplots(n_rows, 6, figsize=(12, n_rows*2), dpi=75)
        ax = ax.flatten()
    for n, col in enumerate(data.columns):
        s = data[col]
        nona = s.dropna()
        scale = scales[n] if scales[n] is not None else helpers.guess_scale(nona)
        res = {'var': col, 'scale': scale}
        if scale == 'binary':
            counts = pd.crosstab(s, np.ones(len(s)))
            if plot:
                comparisons._plot_bars(counts.T, ax[n])
            count = counts.iloc[:,0]
            res['description'] = f'{count.iloc[-1]}/{count.sum()} ({count.iloc[-1] / count.sum() * 100:2.0f}%)'
        elif scale == 'categorical':
            # na values not shown nor reported
            counts = pd.crosstab(s, np.ones(len(s)), dropna=True)
            if plot:
                comparisons._plot_bars(counts.T, ax[n])
            count = counts.iloc[:,0]
            total = count.sum()
            res_list = ([f'{label}: {value}/{total} ({value / count.sum() * 100:.0f}%)' for label, value in count.items()])
            res['description'] = ', '.join(res_list)
        elif scale == 'categorical' or scale == 'continuous':
            p, lp = comparisons.test_for_normality(nona)
            possibly_normal = p > ALPHA
            possibly_lognormal = lp > ALPHA
            if (not possibly_normal) and possibly_lognormal:
                warnings.warn(f'Variable "{col}" might have lognormal distribution.')
            if plot:
                comparisons._plot_histograms(nona, [nona], [possibly_normal], [possibly_lognormal], [ax[n]])
            if possibly_normal or parametric:
                res['description'] = helpers.format_float(np.mean(s)) + ' ± ' + helpers.format_float(np.std(s, ddof=1))
            else:
                sum5n = np.percentile(nona, [0, 25, 50, 75, 100], method='midpoint')
                res['description'] = [helpers.format_float(quantile) for quantile in sum5n]
                # res['description'] = f'{helpers.format_float(sum5n[2])} ({helpers.format_float(sum5n[1])}, {helpers.format_float(sum5n[3])})'

        if plot:
            ax[n].set_title(col)
        results.append(res)
    if plot:
        fig.tight_layout()
    return pd.DataFrame(results)
