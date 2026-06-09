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

def describe(data, parametric=None, scales=None, plot=True, summary='5 numbers'):
    """Generates descriptive statistics for variables in a dataset.

    This function analyzes each column of a pandas DataFrame or Series,
    determines its scale (binary, categorical, or continuous), and computes
    appropriate summary statistics. It can also generate plots to visualize
    the distribution of each variable.

    Args:
        data (pd.DataFrame or pd.Series): Input data to analyze.
        parametric (bool or iterable, optional): Force parametric summary (mean ± std)
            for continuous variables. If an iterable, it's applied per variable.
            Defaults to None, which relies on a normality test.
        scales (str or list, optional): Specify the scale for each variable
            ('binary', 'categorical', 'continuous'). If a list, it's applied
            per variable. Defaults to None (auto-detects scale).
        plot (bool, optional): If True, creates a plot for each variable.
            Defaults to True.
        summary (str, optional): For non-normal data the options are '5 numbers', 
            'median (IQR)' or 'median (range)'. Defaults to '5 numbers'.

    Returns:
        pd.DataFrame: A DataFrame with summary statistics for each variable,
            containing the columns 'var', 'scale', and 'description'.
    """    
    results = []
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    if type(scales) != list:
        scales = [scales] * data.shape[1]
    if hasattr(parametric, '__iter__') and type(parametric) != str:
        parametric = list(parametric) + [None] * (len(data) - len(parametric))
    else:
        parametric = [parametric] * len(data)

    if plot:
        n_rows = ceil(data.shape[1] / 6)
        fig, ax = plt.subplots(n_rows, 6, figsize=(12, n_rows*2), dpi=75)
        ax = ax.flatten()
    for n, (col, prmt) in enumerate(zip(data.columns, parametric)):
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
        elif scale == 'continuous':
            p, lp = comparisons.test_for_normality(nona)
            possibly_normal = p > ALPHA
            possibly_lognormal = lp > ALPHA
            if (not possibly_normal) and possibly_lognormal:
                warnings.warn(f'Variable "{col}" might have lognormal distribution.')
            if plot:
                comparisons._plot_histograms(nona, [nona], [possibly_normal], [possibly_lognormal], [ax[n]])
            if prmt or (possibly_normal and prmt is None):
                res['description'] = helpers.format_value(s.mean()) + ' ± ' + helpers.format_value(s.std())
            else:
                res['description'] = helpers.get_summary(nona, summary)
        if plot:
            ax[n].set_title(col)
        results.append(res)
    if plot:
        fig.tight_layout()
        for n in range(n+1, n_rows*6):
            ax[n].axis('off')
    return pd.DataFrame(results)
