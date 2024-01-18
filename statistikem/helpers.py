import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import re

def guess_scale(var):
    unq = pd.unique(var)
    n_unq = np.sum(~ pd.isna(unq))
    if n_unq <= 2:
        return 'binary'
    elif n_unq <= np.sqrt(var.size):
        return 'categorical'
    elif var.dtype in [float, int, np.integer]:
        return 'continuous'
    else:
        raise ValueError('Variable type not recognized.')

def fix_column_names(df):
    # transtable = ''.maketrans(' .', '__')
    regex = re.compile(r'\W+')
    df.columns = [regex.sub('_', col).strip('_')  for col in df.columns]

def ci_mean(series, level=0.95):
    mean = series.mean()
    q = (level + 1) / 2
    hci = series.sem() * stats.t.ppf(q, len(series))
    return {'mean' : mean, 'min' : mean - hci, 'max' : mean + hci}

# Ulf Olsson (2005) Confidence Intervals for the Mean of a Log-Normal Distribution, 
# Journal of Statistics Education, 13:1
def ci_mean_lognormal(x, level=0.95):
    n = len(x)
    y = np.log(x)
    var_y = np.var(y, ddof=1)
    ln_mean = np.mean(y) + var_y / 2
    q = (level + 1) / 2
    ln_sem = np.sqrt(var_y / n + var_y**2 / 2 / (n - 1))
    hci = stats.t.ppf(q, len(y)) * ln_sem
    return {'mean' : np.exp(ln_mean), 
            'min'  : np.exp(ln_mean - hci), 
            'max'  : np.exp(ln_mean + hci)}

def format_p(p, style='NEJM'):
    """
    According to NEJM statistical guidelines for authors (A.1.g):
    In general, P values larger than 0.01 should be reported to two decimal 
    places, and those between 0.01 and 0.001 to three decimal places; 
    P values smaller than 0.001 should be reported as P<0.001. 
    Notable exceptions to this policy include P values arising from tests 
    associated with stopping rules in clinical trials or from genomewide 
    association studies.
    """
    if hasattr(p, '__iter__') and type(p) != str:
        return [format_p(p_i) for p_i in p]
    if pd.isna(p):
        return ''
    if p > 1.0:
        raise ValueError(f'P value cannot be > 1.0. Received {p:.2f}.')
    if style == 'NEJM':
        if p == 1.0: return '1.0'
        elif p > 0.01: return f'{p:.2f}'
        elif p > 0.001: return f'{p:.3f}'
        else: return '<0.001'
    else:
        if p == 1.0:
            return '1.0'
        elif p < 0.001:
            return '<0.001'
        else:
            return f'{p:.{2 if p > 0.2 else 3}f}'

def format_float(x, precision=2):
    if 1 <= x < 10000:
        return f'{x:#.{precision}f}'
    else:
        return f'{x:#.{precision}g}'

def stars(p):
    if hasattr(p, '__iter__') and type(p) != str:
        return [stars(p_i) for p_i in p]
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''
def plot_table(cells, style=None, global_style=None, colWidths=None,
               rowHeights=None, loc=None, bbox=None, ax=None, **kwa):
    if not ax:
        fig, ax = plt.subplots()
    if global_style is None:
        if mpl.rcParams['axes.facecolor'] == '#E5E5E5':
            global_style = 'modern'
        else:
            global_style = 'oldschool'
    if loc is None or loc == 'full':
        loc, bbox = None, [0, 0, 1, 1]
        
    # Now create the table
    table = mpl.table.Table(ax, loc, bbox, **kwa)
    height = table._approx_text_height() * 1.2

    cells = np.array(cells, dtype=object)
    nrows, ncols = cells.shape

    style = np.array(style, dtype=object)
    if style.ndim == 0:
        style = style[None, None]
    if style.ndim == 1:
        pass
    elif style.ndim == 2 and style.shape[1] == 1 and ncols != 1:
        style = np.repeat(style, ncols, axis=1)
#     return style
#     print(style)

    styles = {
        'bold' : {'fontproperties' : {'weight' : 'bold'}},
        'normal' : {'fontproperties' : {'weight' : 'normal'}},
        'center' : {'loc' : 'center'},
        'left' : {'loc' : 'left'},
        'right' : {'loc' : 'right'},
        'open' : {'edges' : 'open'},
        'closed' : {'edges' : 'closed'}
    }
        
    if colWidths is None:
        colWidths = [1.0 / ncols] * ncols

    # Add the cells
    for rn in range(cells.shape[0]):
        for cn in range(cells.shape[1]):
            ckw = {}
            
            # format cell contents
            cell = cells[rn, cn]
            if cell is None:
                text = ''
            elif isinstance(cell, str):
                text = cell
                ckw['loc'] = 'left'
            elif isinstance(cell, float):
                text = format_float(cell)
                ckw['loc'] = 'right'
            elif isinstance(cell, (int, np.integer)):
                text = str(cell)
                ckw['loc'] = 'right'
            else:
                text = str(cell)
                ckw['loc'] = 'left'
                
            # apply default style (this should eventually move to Cell)
            if global_style == 'modern':
                ckw['edgecolor'] = 'white'
                if rn == 0:
                    ckw['fontproperties'] = {'weight' : 'bold'}
                    ckw['loc'] = 'center'
                    ckw['facecolor'] = 'silver'
                elif rn % 2:
                    ckw['facecolor'] = 'whitesmoke'
                else:
                    ckw['facecolor'] = 'gainsboro'
            else:
                if rn == 0:
                    ckw['fontproperties'] = {'weight' : 'bold'}
                    ckw['loc'] = 'center'
            
            # apply local style
            if rn < style.shape[0] and cn < style.shape[1] and style[rn, cn]:
                cs = style[rn, cn]
                if isinstance(cs, str):
                    for key in cs.split():
                        if key.startswith('fc_'):
                            ckw['facecolor'] = key[3:]
                        else:
                            ckw.update(styles[key])
                elif isinstance(cs, dict):
                    ckw.update(cs)
            table.edges = ckw.pop('edges', None)
            table.add_cell(rn, cn,
                           width=colWidths[cn], 
                           height=height,
                           text=text,
                            **ckw)

    ax.add_table(table)
#     table.scale(1, 1.5)
#     ax.set_facecolor('pink')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')
    return table
