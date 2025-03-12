import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import re
from dotenv import load_dotenv
import os
import jaydebeapi
import warnings
from getpass import getpass

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
        raise ValueError(f'Variable `{var.name}` type not recognized.')

def fix_column_names(df):
    # transtable = ''.maketrans(' .', '__')
    regex = re.compile(r'\W+')
    df.columns = [regex.sub('_', col).strip('_')  for col in df.columns]

def ci_mean_normal(series, level=0.95):
    """Calculate parametrically mean and CI of a normally distributed variable.

    Parameters
    ----------
    x : array or Series
        Data.
    level : float
        Confidence level. Default 0.95.

    Returns
    -------
    (mean, min, max)
    """
    mean = series.mean()
    q = (level + 1) / 2
    hci = series.sem() * stats.t.ppf(q, len(series))
    return mean, mean - hci, mean + hci

# Ulf Olsson (2005) Confidence Intervals for the Mean of a Log-Normal Distribution, 
# Journal of Statistics Education, 13:1
def ci_mean_lognormal(x, level=0.95):
    """Calculate parametrically mean and CI of a lognormally distributed variable.

    According to:
    Ulf Olsson (2005) Confidence Intervals for the Mean of a Log-Normal Distribution, 
    Journal of Statistics Education, 13:1
    
    Parameters
    ----------
    x : array or Series
        Data.
    level : float
        Confidence level. Default 0.95.

    Returns
    -------
    (mean, min, max)
    """
    n = len(x)
    y = np.log(x)
    var_y = np.var(y, ddof=1)
    ln_mean = np.mean(y) + var_y / 2
    q = (level + 1) / 2
    ln_sem = np.sqrt(var_y / n + var_y**2 / 2 / (n - 1))
    hci = stats.t.ppf(q, len(y)) * ln_sem
    return np.exp(ln_mean), np.exp(ln_mean - hci), np.exp(ln_mean + hci)

def ci_mean_bootstrap(s, level=0.95, as_df=True):
    """Calculate mean and CI by bootstrap.

    Parameters
    ----------
    x : array or Series
        Data.
    level : float
        Confidence level. Default 0.95.

    Returns
    -------
    (mean, min, max)
    """
    bs = stats.bootstrap((s.dropna(),), np.mean, confidence_level=.95, n_resamples=10_000)
    mean = bs.bootstrap_distribution.mean()
    ci_l, ci_h = bs.confidence_interval.low, bs.confidence_interval.high
    res = pd.DataFrame([{'mean':mean, 'ci_l':ci_l, 'ci_h':ci_h}]) if as_df else mean, ci_l, ci_h
    return res

def format_p(p, style=None):
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


def _get_series(var, df):
    if var is None:
        return None
    elif type(var) == pd.Series:
        return var
    elif df is not None and var in df.columns:
        return df[var]
    elif df is not None:
        raise ValueError(f'"{var}" not found in the dataframe.')
    else:
        raise ValueError(f'Dataframe not passed.')

def highlight(string, pattern):
    if type(pattern) == list:
        pattern = '(' + ')|('.join(pattern) + ')'
    start = "\033[31m"
    end = "\033[0m"
    regex = re.compile(pattern, re.IGNORECASE)
    matches = regex.finditer(string)
    s = string
    offset = 0
    for match in matches:
        start_index = match.start() + offset
        end_index = match.end() + offset
        s = (s[:start_index] + start + s[start_index:end_index] + end + s[end_index:])
        offset += len(start) + len(end)
    return s

def read_sql(sql, server='jdbc_analytics', parse_dates=None, url=None, user=None, password=None, **kwargs):
    load_dotenv()
    env = os.getenv(server.upper())
    env = [] if env is None else env.split(' ')
    if url is None:
        url = env[0] if len(env) > 0 else input('url:')
    if user is None:
        user = env[1] if len(env) > 1 else input('username:')
    password = env[2] if len(env) > 2 else getpass('password:')
    driver = os.getenv('JDBC_DRIVER')
    with jaydebeapi.connect(driver, url, [user, password]) as connection:
        with warnings.catch_warnings(action='ignore'):
            return pd.read_sql_query(sql, connection, parse_dates=parse_dates, **kwargs)

