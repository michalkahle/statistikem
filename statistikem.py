import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import statsmodels.api as sm
from scipy import stats
import warnings
import re

FIGSIZE = (10, 2)
WIDTH_RATIOS=(4,2,2,2)
FONTSIZE = 10
ALPHA = 0.05

def univariate_tests(var_list, grouping, df, plot=True, scales={}, 
                     filter=None, sort=None, **kwa):
    ll = []
    orig_mow = plt.rcParams['figure.max_open_warning'] = 0
    for varname in var_list:
        var = df[varname]
        scale = scales.get(varname, _guess_scale(var))
        if not filter or filter == scale:
            res = univariate_test(var, df[grouping], plot=plot, scale=scale, **kwa)
            ll.append(res)
    plt.rcParams['figure.max_open_warning'] = orig_mow
    res_df = pd.DataFrame(ll)
    if sort:
        res_df = res_df.sort_values(sort)
    return res_df

def univariate_test(var, grouping, df=None, plot=True, scale=None, **kwa):
    if type(var) != pd.Series:
        if df is not None:
            var = df[var]
        else:
            raise ValueError('`var` should be a Series. Alternatively `df` should be passed.')
    if type(grouping) != pd.Series:
        if df is not None:
            grouping = df[grouping]
        else:
            raise ValueError('`grouping` should be a Series. Alternatively `df` should be passed.')
    if not scale:
        scale = _guess_scale(var)
    res = dict(var=var.name, scale=scale)
    if scale == 'binary':
        res = univariate_frequency_test(var, grouping, plot=plot, res=res, **kwa)
    elif scale == 'categorical':
        res = univariate_location_test(var, grouping, plot=plot, res=res, scale='categorical', **kwa)
    elif scale == 'continuous':
        res = univariate_location_test(var, grouping, plot=plot, res=res, scale='continuous', **kwa)
    else:
        raise Exception(f'Unknown scale: {scale}')
    return res


def univariate_location_test(var, grouping, plot=True, res={}, scale=None, **kwa):
    na = var.isna()
    var_nona = var[~ na]
    grp_nona = grouping[~ na]
    g_names, gg, g_missing = _split_to_groups(var_nona, na, grouping, grp_nona)
    n_groups = len(gg)
        
    tests = [['test', 'p-value']]
    tests_style = [['bold center'] * 2]
    
    # Shapiro-Wilk test for normal distribution in groups
    p_shapiro, p_logshapiro = [], []
    for group in gg:
        if len(group) < 3 or np.ptp(group) == 0:
            p, lp = 0, 0
        else:
            _, p = stats.shapiro(group)
            _, lp = stats.shapiro(np.log(group)) if group.min() > 0 else (0, 0)
        p_shapiro.append(p)
        p_logshapiro.append(lp)
    possibly_normal = np.array(p_shapiro) > ALPHA
    possibly_lognormal = np.array(p_logshapiro) > ALPHA
    
    if np.all(possibly_normal):
        distribution = 'normal'
    elif np.all(possibly_lognormal):
        distribution = 'lognormal'
    else:
        distribution = None
        
    if not all(possibly_normal) and all(possibly_lognormal):
        warnings.warn(f'{var.name}: all groups possibly lognormal. Tests not implemented, yet!')
    
    
    
    
    
    
    if n_groups == 1:
        raise NotImplementedError('Just one group.')
    elif n_groups == 2:
        # Levene test for equal variances
        center = 'mean' if np.all(possibly_normal) else 'median'
        s_levene, p_levene = stats.levene(*gg, center=center)
        equal_var = p_levene > ALPHA
        tests.append(['Levene', p_levene])
        tests_style.append(['', '' if equal_var else 'fc_pink'])
        
        # t-test for the means of two independent samples
        s_t, p_t = stats.ttest_ind(*gg, equal_var=equal_var)
        tests.append(['Student\'s t' if equal_var else 'Welch\'s t', p_t])
        tests_style.append(['', 'fc_pink' if p_t < ALPHA else ''])
        
        # Mann-Whitney U test
        s_mw, p_mw = stats.mannwhitneyu(gg[0], gg[1], alternative='two-sided', use_continuity=False)
        tests.append(['Mann-Whitney', p_mw])
        tests_style.append(['', 'fc_pink' if p_mw < ALPHA else ''])
        
        if np.all(possibly_normal):
            res['test'] = 't'
            res['p'] = p_t
        else:
            res['test'] = 'Mann-Whitney'
            res['p'] = p_mw
    else:
        # ANOVA
        pass
    
    if plot:
        fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
        spec = fig.add_gridspec(1, 4, width_ratios=WIDTH_RATIOS)
        
        ax0 = fig.add_subplot(spec[0,0])
        ax0.set_title(f'{var.name} \ {grouping.name}', loc='left')
        table = [
            [None] + list(g_names) + ['total'],
            ['n'] + [len(g) for g in gg] + [len(var_nona)],
            ['missing'] + list(g_missing) + [na.sum()],
            ['median'] + [np.median(g) for g in gg] +[np.median(var_nona)],
            ['mean'] + [np.mean(g) for g in gg] + [np.mean(var_nona)],
            ['SD'] + [np.std(g) for g in gg] + [np.std(var_nona)],
        ]
        style = [[None] * len(table[0])] * len(table)
#         style = [[None] + [f'fc_C{x}' for x in range(n_groups)] + [None]]
        style[0] = [None] + [f'fc_C{x}' for x in range(n_groups)] + [None]
        style[4] = [None] + ['fc_lightgreen' if x else '' for x in possibly_normal] + [None]
        #colWidths=[2,2,2,2]
        table_artist = plot_table(table, style=style, loc='full', ax=ax0, )
        table_artist.auto_set_font_size(False)
        table_artist.set_fontsize(FONTSIZE)
        
        ax1 = fig.add_subplot(spec[0,1])
        ax1.set(title=scale)
        if scale == 'continuous':
            _, loc, _ = ax1.hist(gg, rwidth=1.0)
            binwidth = loc[1] - loc[0]
            X = np.linspace(var_nona.min(), var_nona.max(), 20)
            for ii, g in enumerate(gg):
                normal, lognormal = possibly_normal[ii], possibly_lognormal[ii]
                if normal:
                    Y = stats.norm.pdf(X, loc=g.mean(), scale=g.std()) * len(g) * binwidth
                    ax1.plot(X, Y, color=f'C{ii}')
        elif scale == 'categorical':
            counts = var_nona.groupby([var_nona,grp_nona]).count().unstack().T
            _plot_bars(counts, ax1)
        else:
            raise Exception(f'unknown scale: {scale}')
        
        ax2 = fig.add_subplot(spec[0,2])
#         for x in range(len(g_names)):
#             sm.qqplot(gg[x], ax=ax01, markerfacecolor='none', markeredgecolor=f'C{x}')
        ax2.set_title('Q-Q normal~sample')
        ax2.get_xaxis().label.set_visible(False)
        ax2.get_yaxis().label.set_visible(False)
        sm.qqplot(var_nona, line='s', ax=ax2, markerfacecolor='none', markeredgecolor='black')
        
        ax3 = fig.add_subplot(spec[0,3])
        table = plot_table(tests, style=tests_style, ax=ax3)
        table.auto_set_font_size(False)
        table.set_fontsize(FONTSIZE)
    return res

def univariate_frequency_test(var, grouping, plot=True, res={}, **kwa):
    na = var.isna()
    var_nona = var[~ na]
    grp_nona = grouping[~ na]
    g_names, gg, g_missing = _split_to_groups(var_nona, na, grouping, grp_nona)
    n_groups = len(gg)
    
    tests = [['test', 'p-value']]
    tests_style = [['bold center'] * 2]
    counts = pd.crosstab(var_nona, grp_nona)
    chi2_valid = '' if counts.values.min() >= 5 else 'fc_pink'

    if var_nona.unique().shape[0] > 1:
        test = 'Pearson chi^2'
        chi2, p, dof, exp = stats.chi2_contingency(counts, correction=False)
        res['test'], res['p'] = test, p
        tests.append([r'$\chi^2$ Pearson', p])
        tests_style.append([chi2_valid, 'fc_pink' if p < ALPHA else ''])

        test = 'Yates chi^2'
        chi2, p, dof, exp = stats.chi2_contingency(counts, correction=True)
        res['test'], res['p'] = test, p
        tests.append([r'$\chi^2$ Yates', p])
        tests_style.append([chi2_valid, 'fc_pink' if p < ALPHA else ''])

        test = 'Fisher exact'
        oddsratio, p = stats.fisher_exact(counts, alternative='two-sided')
        res['test'], res['p'] = test, p
        tests.append([test, p])
        tests_style.append(['', 'fc_pink' if p < ALPHA else ''])
        tests.append(['odds ratio', oddsratio])
        tests_style.append([None, None])
    
    if plot:
        fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
        spec = fig.add_gridspec(1, 4, width_ratios=WIDTH_RATIOS)

        ax0 = fig.add_subplot(spec[0,0])
        ax0.set_title(f'{var.name} \ {grouping.name}', loc='left')
        tdata = [[''  ,   g_names[0], g_names[1], 'total']]
        tstyle = [[None, 'fc_C0', 'fc_C1', 'normal']]
        warn = lambda x: 'fc_pink' if x < 5 else ''
        sums = counts.sum()
        total = sums.sum()
        perc = lambda x: f'{x} ({x / total * 100:2.0f}%)'
        for val, row in counts.iterrows():
            tdata.append([val, perc(row[0]), perc(row[1]), perc(row.sum())])
            tstyle.append(['', 'right ' + warn(row[0]), 'right ' + warn(row[1]), 'right'])
        tdata.append(['total', perc(sums[0]), perc(sums[1]), perc(total)])
        tstyle.append(['', 'right ', 'right', 'right'])
        table = plot_table(tdata, style=tstyle, loc='full', ax=ax0, colWidths=[1,1,1,1])
        table.auto_set_font_size(False)
        table.set_fontsize(FONTSIZE)

        ax1 = fig.add_subplot(spec[0,1])
        _plot_bars(counts.T, ax1)

        ax2 = fig.add_subplot(spec[:,2])
        ax2.set_title('Observed vs Expected')
        sums = counts.sum(axis=1)
        if sums.shape[0] > 1:
            ax2.plot([0, sums[0]], [0, sums[1]], color='black')
            for ii, col in counts.T.iterrows():
                ax2.plot(col[0], col[1], 'o')
            ax2.set_aspect('equal', adjustable='box')
            ax2.set_xlabel(sums.index[0])
            ax2.set_ylabel(sums.index[1])

        ax3 = fig.add_subplot(spec[:,3])
        table = plot_table(tests, style=tests_style, ax=ax3)
        table.auto_set_font_size(False)
        table.set_fontsize(FONTSIZE)
    return res







def plot_table(cells, style=None, global_style=None, 
               colWidths=None, rowHeights=None,
               loc=None, bbox=None, ax=None, **kwargs):
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
    table = mpl.table.Table(ax, loc, bbox, **kwargs)
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
                text = f'{cell:#.2f}' if 1 <= cell < 10000 else f'{cell:#.2g}'
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

def _plot_bars(counts, ax):
    n_groups = counts.shape[0]
    w_total = 0.8
    w_single = w_total / n_groups
    X_base = np.arange(counts.shape[1])
    for ii in range(n_groups):
        group = counts.iloc[ii]
        X = X_base - w_total/2 + ii*w_single
        ax.bar(X, group, width=w_single, align='edge')
    ax.set_xticks(X_base)
    ax.set_xticklabels(group.index)
    
def _guess_scale(var):
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

def _split_to_groups(var_nona, na, grouping, grp_nona):
    if isinstance(grp_nona, pd.Series) and grp_nona.dtype.name == 'category':
        g_names = grp_nona.cat.categories
        gg = [var_nona[grp_nona == name] for name in g_names]
        g_missing = [na[grouping == name].sum() for name in g_names]
    else:
        glu = np.sort(grp_nona.unique())
        g_names = [str(g) for g in glu]
        gg = [var_nona[grp_nona == g] for g in glu]
        g_missing = [na[grouping == g].sum() for g in glu]
    return g_names, gg, g_missing

def ci_mean(series, level=0.95):
    mean = series.mean()
    q = (level + 1) / 2
    hci = series.sem() * stats.t.ppf(q, len(series))
    return {'mean' : mean, 'min' : mean - hci, 'max' : mean + hci}

# Ulf Olsson (2005) Confidence Intervals for the Mean of a Log-Normal Distribution, Journal of Statistics Education, 13:1
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
