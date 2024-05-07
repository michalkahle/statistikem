import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import statsmodels.api as sm
from scipy import stats
import warnings
import re
import seaborn as sns

from .helpers import format_p
from .helpers import format_float

p_cmap = mpl.colors.LinearSegmentedColormap('p_cmap', {
     'red':   [(0.0,  0.0, 0.0),
               (0.1,  1.0, 1.0),
               (1.0,  1.0, 1.0)],

     'green': [(0.0,  0.0, 0.0),
               (0.05, 1.0, 1.0),
               (1.0,  1.0, 1.0)],

     'blue':  [(0.0,  0.0, 0.0),
               (0.1,  1.0, 1.0),
               (1.0,  1.0, 1.0)]})

def correlate(rows, cols=None, kind='Spearman', data=None, plot=True):
    cols = rows if cols is None else cols
    rr = np.zeros([rows.shape[1], cols.shape[1]])
    pp = rr.copy()
    na = rr.copy()
    for row_n, (_, row) in enumerate(rows.items()):
        for col_n, (_, col) in enumerate(cols.items()):
            nona = row.notna() & col.notna()
            if kind == 'Pearson':
                r, p = stats.pearsonr(row[nona], col[nona])
            elif kind == 'Spearman':
                r, p = stats.spearmanr(row[nona], col[nona])
            elif kind == 'Kendall':
                r, p = stats.kendalltau(row[nona], col[nona])
            else:
                r, p = 0, 0
            rr[row_n, col_n] = r
            pp[row_n, col_n] = p
    
    rr = pd.DataFrame(rr, index=rows.columns, columns=cols.columns)
    pp = pd.DataFrame(pp, index=rows.columns, columns=cols.columns)
    if plot:
        if rr.shape[1] < 15:
            fig, ax = plt.subplots(
                nrows=1, ncols=2, 
                figsize=(rr.shape[1]*1.2+2, rr.shape[0]/2+1), 
                sharey=rr.shape[1] < 5,
                dpi=75
            )
        else:
            fig, ax = plt.subplots(
                nrows=2, ncols=1, 
                figsize=(rr.shape[1]/2+1, rr.shape[0]*.8+2.5), 
                sharex=rr.shape[0] < 5,
                dpi=75
            )
        # seaborn should be replaced by a helper function:
        # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
        sns.heatmap(rr, cmap='coolwarm', center=0, cbar=False, annot=True, fmt='.2f', ax=ax[0])
        ax[0].set_title(f'{kind} correlations')
        sns.heatmap(pp, cmap=p_cmap, cbar=False, annot=True, ax=ax[1]) #'pink'
        ax[1].set_title(f'p-values')
        fig.tight_layout()
    return rr, pp

def plot_correlation(var1, var2, kind='Pearson', data=None, xlabel=None, ylabel=None, ploc=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
    var1 = data[var1] if type(var1) == str else var1
    var2 = data[var2] if type(var2) == str else var2
    nona = var1.notna() & var2.notna()
    if kind == 'Pearson':
        r, p = stats.pearsonr(var1[nona], var2[nona])
        symbol = 'r'
    elif kind == 'Spearman':
        r, p = stats.spearmanr(var1[nona], var2[nona])
        symbol = 'ϱ'
        
    if ploc is None:
        ploc = 'lower right' if r > 0 else 'lower left'
    ax.scatter(var1, var2, **kwargs)
    p = format_p(p)
    p = p if '<' in p else '= ' + p
    at = mpl.offsetbox.AnchoredText(f'{symbol} = {r:.2f}\np {p}', loc=ploc, frameon=False)
    ax.add_artist(at)
    ax.set_xlabel(var1.name if xlabel is None else xlabel)
    ax.set_ylabel(var2.name if ylabel is None else ylabel)
    ax.get_figure().tight_layout()