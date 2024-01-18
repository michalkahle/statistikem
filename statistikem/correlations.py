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

def correlate(rows, cols=None, type='Spearman', data=None, plot=True):
    cols = rows if cols is None else cols
    rr = np.zeros([rows.shape[1], cols.shape[1]])
    pp = rr.copy()
    na = rr.copy()
    for row_n, (_, row) in enumerate(rows.items()):
        for col_n, (_, col) in enumerate(cols.items()):
            nona = row.notna() & col.notna()
            if type == 'Pearson':
                r, p = stats.pearsonr(row[nona], col[nona])
            elif type == 'Spearman':
                r, p = stats.spearmanr(row[nona], col[nona])
            elif type == 'Kendall':
                r, p = stats.kendalltau(row[nona], col[nona])
            else:
                r, p = 0, 0
            rr[row_n, col_n] = r
            pp[row_n, col_n] = p
    
    rr = pd.DataFrame(rr, index=rows.columns, columns=cols.columns)
    pp = pd.DataFrame(pp, index=rows.columns, columns=cols.columns)
    if plot:
        if rr.shape[1] < 15:
            nrows, ncols, figsize = 1, 2, (rr.shape[1]*1.2+1, rr.shape[0]/2+1)
        else:
            nrows, ncols, figsize = 2, 1, (rr.shape[0]/2+1, rr.shape[1]*.8+1)
            
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, dpi=75)
        sns.heatmap(rr, cmap='coolwarm', center=0, cbar=True, annot=True, fmt='.2f', ax=ax[0])
        ax[0].set_title(f'{type} correlations')
        sns.heatmap(pp, cmap=p_cmap, cbar=True, annot=True, ax=ax[1]) #'pink'
        ax[1].set_title(f'{type} correlations [p-values]')
        fig.tight_layout()
    return rr, pp
