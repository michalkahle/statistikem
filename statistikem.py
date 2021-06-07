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
    if df is not None:
        varname = var
        # groupname = grouping
        var = df[var]
        grouping = df[grouping]
    else:
        varname = var.name
    if not scale:
        scale = _guess_scale(var)
    res = dict(var=varname, scale=scale)
    if scale == 'binary':
        res = univariate_frequency_test(var, grouping, plot=plot, res=res, **kwa)
    elif scale == 'categorical':
        res = univariate_location_test(var, grouping, plot=plot, res=res, scale='categorical', **kwa)
    elif scale == 'continuous':
        res = univariate_location_test(var, grouping, plot=plot, res=res, scale='continuous', **kwa)
    else:
        raise Exception(f'Unknown scale: {scale}')
    return res

def _split_to_groups(data, na, grouping, labels):
    if isinstance(labels, pd.Series) and labels.dtype.name == 'category':
        g_names = labels.cat.categories
        gg = [data[labels == name] for name in g_names]
        g_missing = [na[grouping == name].sum() for name in g_names]
    else:
        glu = np.sort(labels.unique())
        if hasattr(grouping, 'name'):
            g_names = [grouping.name + ':' + str(g) for g in glu]
        else:
            g_names = ['group' + ':' + str(g) for g in glu]
        gg = [data[labels == g] for g in glu]
        g_missing = [na[grouping == g].sum() for g in glu]
    return g_names, gg, g_missing


def univariate_location_test(var, grouping, plot=True, res={}, scale=None, **kwa):
    na = var.isna()
    data = var[~ na]
    labels = grouping[~ na]
    g_names, gg, g_missing = _split_to_groups(data, na, grouping, labels)
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
        warnings.warn('All groups possibly lognormal. Tests not implemented, yet!')
    
    
    
    
    
    
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
        ax0.set_title(var.name, loc='left')
        table = [
            [None] + list(g_names) + ['Total'],
            ['n'] + [len(g) for g in gg] + [len(data)],
            ['missing'] + list(g_missing) + [na.sum()],
            ['median'] + [np.median(g) for g in gg] +[np.median(data)],
            ['mean'] + [np.mean(g) for g in gg] + [np.mean(data)],
            ['SD'] + [np.std(g) for g in gg] + [np.std(data)],
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
            X = np.linspace(data.min(), data.max(), 20)
            for ii, g in enumerate(gg):
                normal, lognormal = possibly_normal[ii], possibly_lognormal[ii]
                if normal:
                    Y = stats.norm.pdf(X, loc=g.mean(), scale=g.std()) * len(g) * binwidth
                    ax1.plot(X, Y, color=f'C{ii}')
        elif scale == 'categorical':
            counts = data.groupby([data,labels]).count().unstack().T
            _plot_bars(counts, ax1)
        else:
            raise Exception(f'unknown scale: {scale}')
        
        ax2 = fig.add_subplot(spec[0,2])
#         for x in range(len(g_names)):
#             sm.qqplot(gg[x], ax=ax01, markerfacecolor='none', markeredgecolor=f'C{x}')
        ax2.set_title('Q-Q normal~sample')
        ax2.get_xaxis().label.set_visible(False)
        ax2.get_yaxis().label.set_visible(False)
        sm.qqplot(data, line='s', ax=ax2, markerfacecolor='none', markeredgecolor='black')
        
        ax3 = fig.add_subplot(spec[0,3])
        table = plot_table(tests, style=tests_style, ax=ax3)
        table.auto_set_font_size(False)
        table.set_fontsize(FONTSIZE)
    return res

def univariate_frequency_test(var, grouping, plot=True, res={}, **kwa):
    na = var.isna()
    data = var[~ na]
    labels = grouping[~ na]
    g_names, gg, g_missing = _split_to_groups(data, na, grouping, labels)
    n_groups = len(gg)
        
    tests = [['test', 'p-value']]
    tests_style = [['bold center'] * 2]
    
    
    
    
    
    
    obs = np.array(
        [[np.sum(group == val) for group in gg] for val in data.unique()])
    chi2_valid = '' if np.min(obs) >= 5 else 'fc_pink'
    
    
    
    
    counts = data.groupby([data,labels]).count().unstack()
    chi2_valid = '' if counts.values.min() >= 5 else 'fc_pink'
    
    
    
    
    

    test = 'Pearson chi^2'
    chi2, p, dof, exp = stats.chi2_contingency(obs, correction=False)
    res['test'], res['p'] = test, p
    tests.append([r'$\chi^2$ Pearson', p])
    tests_style.append([chi2_valid, 'fc_pink' if p < ALPHA else ''])

    test = 'Yates chi^2'
    chi2, p, dof, exp = stats.chi2_contingency(obs, correction=True)
    res['test'], res['p'] = test, p
    tests.append([r'$\chi^2$ Yates', p])
    tests_style.append([chi2_valid, 'fc_pink' if p < ALPHA else ''])

    test = 'Fisher exact'
    oddsratio, p = stats.fisher_exact(obs, alternative='two-sided')
    res['test'], res['p'] = test, p
    tests.append([test, p])
    tests_style.append(['', 'fc_pink' if p < ALPHA else ''])
    tests.append(['odds ratio', oddsratio])
    tests_style.append([None, None])
    
    
    if plot:
        fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
        spec = fig.add_gridspec(1, 4, width_ratios=WIDTH_RATIOS)

        ax0 = fig.add_subplot(spec[0,0])
        ax0.set_title(var.name, loc='left')
        tdata = [[''  ,   g_names[0], g_names[1], 'Total']]
        tstyle = [[None, 'fc_C0', 'fc_C1', 'normal']]
        warn = lambda x: 'fc_pink' if x < 5 else ''
        for ii, val in enumerate(data.unique()):
            tdata.append([val, obs[ii,0], obs[ii,1], obs[ii,:].sum()])
            tstyle.append(['', warn(obs[ii,0]), warn(obs[ii,1]), ''])
        tdata.append(['Total', obs[:,0].sum(), obs[:,1].sum(), obs.sum()])
        table = plot_table(tdata, style=tstyle, loc='full', ax=ax0, colWidths=[1,1,1,1])
        table.auto_set_font_size(False)
        table.set_fontsize(FONTSIZE)

        ax1 = fig.add_subplot(spec[0,1])
        counts = data.groupby([data,labels]).count().unstack().T
        _plot_bars(counts, ax1)

        ax2 = fig.add_subplot(spec[:,2])
        ax2.set_title('Observed vs Expected')
        counts = data.groupby([data,labels]).count().unstack().T
        sums = counts.sum()
        ax2.plot([0, sums[0]], [0, sums[1]], color='black')
        for ii, group in counts.iterrows():
            ax2.plot(group[0], group[1], 'o')
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
            if isinstance(cell, str):
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
