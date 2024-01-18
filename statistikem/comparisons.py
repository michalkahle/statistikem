import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import statsmodels.api as sm
from scipy import stats
import warnings
import re

from .helpers import guess_scale
from .helpers import fix_column_names
from .helpers import ci_mean
from .helpers import ci_mean_lognormal
from .helpers import format_p
from .helpers import format_float
from .helpers import stars
from .helpers import plot_table

FONTSIZE = 10
ALPHA = 0.05

def compare(variables, grouping=None, data=None, plot=True, scale=None, 
            sort=None, parametric=None, multi_test_corr='bonferroni', summary=False, **kwa):
    """Perform univariate analysis of one or more variables
    
    Parameters
    ----------
    variables : str, list of str
        Name or list of names of DataFrame columns to be analyzed.
    grouping : str
        Name of column that will be used for grouping of observations.
    data : DataFrame
        A DataFrame containing all analyzed variables and grouping variable.
    plot : bool, default=True
        Plot a graphical summary.
    summary : bool, default=False
        Include summary of the groups in output. Useful for Table 1 creation.
    scale : {'binary', 'categorical', 'continuous'}, optional
        Scale of the variable (for plots). If not given it is guessed from data.
    parametric : bool, optional
        Force parametric or non-parametric tests. By default it is decided
        by the Shapiro-Wilk test of normality.
    multi_test_corr : str, default='bonferroni'
        Method of multiple testing correction of the p-values. Possibilities 
        include: 'bonferroni', 'sidak', 'holm-sidak', 'holm' and other. Please
        see documentation to `statsmodels.stats.multitest.multipletests`.
    
    Returns
    -------
    DataFrame with results of the comparisons containing columns:
        formula : R-style description of the model
        scale : scale of the variable
        test : the statistical test used
        p : the p-value
        If the `summary` parameter is true, all groups and their summaries are
            included. This is useful for Table 1 creation of clinical trials.
    """
    if type(variables) == str:
        variables = [variables]
    if hasattr(scale, '__iter__') and type(scale) != str:
        scale = list(scale) + [None] * (len(variables) - len(scale))
    else:
        scale = [scale] * len(variables)
    if hasattr(parametric, '__iter__') and type(parametric) != str:
        parametric = list(parametric) + [None] * (len(variables) - len(parametric))
    else:
        parametric = [parametric] * len(variables)
    ll = []
    orig_mow = plt.rcParams['figure.max_open_warning'] = 0
    for var, sc, par in zip(variables, scale, parametric):
        if var != grouping:
            res = compare_one(var, grouping, data=data, plot=plot, scale=sc, 
                              parametric=par, summary=summary, **kwa)
            ll.append(res)
    plt.rcParams['figure.max_open_warning'] = orig_mow
    res_df = pd.DataFrame(ll)
    p_corr = sm.stats.multipletests(res_df['p'], method=multi_test_corr)[1]
    res_df['p_corr'] = p_corr
    if sort:
        res_df = res_df.sort_values(sort)
    return res_df

def compare_one(var, grouping=None, data=None, plot=True, summary=False, 
                scale=None, parametric=None, **kwa):
    """Describe and compare one grouped variable
    
    Parameters
    ----------
    var : str, list of str, Series or DataFrame
        Name or list of names of DataFrame columns, Series or DataFrame with
        data. In case of single column or Series the observations are assumed
        to be independent. In case of multiple columns repeated or related
        observations are assumed.
    grouping : str or Series
        Name of column or a Series that will be used for grouping independent
        observations.
    data : DataFrame, optional
        If names of columns are given for `var` and `grouping` then DataFrame
        containing these columns must be given.
    plot : bool, default=True
        Plot graphical summary.
    summary : bool, default=False
        Include summary of the groups in output. Useful for Table 1 creation.
    scale : {'binary', 'categorical', 'continuous'}, optional
        Scale of the variable (for plots). If not given it is guessed from data.
    parametric : bool, optional
        Force parametric or non-parametric tests. By default it is decided
        by the Shapiro-Wilk test of normality.
    
    Returns
    -------
    dict
        result of comparison:
        formula : R-style description of the model
        scale : scale of the variable
        test : the statistical test used
        p : the p-value
        If the `summary` parameter is true, all groups and their summaries are
            included. This is useful for Table 1 creation of clinical trials.
"""
    if type(var) == list:
        var = data[var]
    if type(var) == pd.DataFrame:
        # paired tests
        if not scale:
            scale = guess_scale(var.values.flatten())
        if scale == 'binary':
            res = paired_proportion_test(var, plot=plot, scale=scale, **kwa)
        elif scale == 'categorical' or scale == 'continuous':
            res = paired_difference_test(var, plot=plot, scale=scale, **kwa)
        else:
            raise Exception(f'Unknown scale: {scale}')
    else:
        # independent tests
        var = _get_series(var, data)
        grouping = _get_series(grouping, data)
        if not scale:
            scale = guess_scale(var)
        if scale == 'binary':
            res = independent_proportion_test(var, grouping, plot=plot, summary=summary, scale=scale, **kwa)
        elif scale == 'categorical' or scale == 'continuous':
            res = independent_difference_test(var, grouping, plot=plot, 
                summary=summary, scale=scale, parametric=parametric, **kwa)
        else:
            raise Exception(f'Unknown scale: {scale}')
    return res

def independent_difference_test(var, grouping, plot=True, summary=False, scale=None, parametric=None, **kwa):
    res = {'formula': f'{var.name} ~ {grouping.name}', 'scale': scale, 'test': None, 'p': np.nan}
    na_loc, var_nona, grp_nona, g_names, gg, g_missing = _split_to_groups(var, grouping)
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
    possibly_normal = np.array(p_shapiro) > ALPHA / n_groups # Bonferroni correction
    possibly_lognormal = np.array(p_logshapiro) > ALPHA / n_groups # Bonferroni correction
    
    if np.all(possibly_normal):
        distribution = 'normal'
    elif np.all(possibly_lognormal):
        distribution = 'lognormal'
        warnings.warn(f'Variable "{var.name}" might have lognormal distribution.')
    else:
        distribution = None
    if parametric is None:
        parametric = True if distribution == 'normal' else False
        
        
    
    if n_groups > 1:
        # Levene test for equal variances
        center = 'mean' if parametric else 'median'
        s_levene, p_levene = stats.levene(*gg, center=center)
        equal_var = p_levene > ALPHA
        tests.append(['Levene', p_levene])
        tests_style.append(['', '' if equal_var else 'fc_pink'])
        
    
    if n_groups == 1:
        # Shapiro-Wilk test of normality
        s_sw, p_sw = stats.shapiro(*gg)
        tests.append(['Shapiro-Wilk', p_sw])
        tests_style.append(['', 'fc_pink' if p_sw < ALPHA else ''])
        #res['test'], res['p'] = 'Shapiro-Wilk', p_sw
        # t-test for the mean of one group
        s_t, p_t = stats.ttest_1samp(*gg, popmean=0)
        tests.append(['one sample t', p_t])
        tests_style.append(['', 'fc_pink' if p_t < ALPHA else ''])
        res['test'], res['p'] = 't', p_t
    elif n_groups == 2:
        # t-test for the means of two independent samples
        s_t, p_t = stats.ttest_ind(*gg, equal_var=equal_var)
        tests.append(['Student\'s t' if equal_var else 'Welch\'s t', p_t])
        tests_style.append(['', 'fc_pink' if p_t < ALPHA else ''])
        
        # Mann-Whitney U test
        s_mw, p_mw = stats.mannwhitneyu(gg[0], gg[1], alternative='two-sided', 
                                        use_continuity=scale=='categorical')
        tests.append(['Mann-Whitney', p_mw])
        tests_style.append(['', 'fc_pink' if p_mw < ALPHA else ''])
        
        if parametric:
            res['test'], res['p'] = 't', p_t
        else:
            res['test'], res['p'] = 'Mann-Whitney', p_mw
    else:
        # ANOVA
        s_a, p_a = stats.f_oneway(*gg)
        tests.append(['ANOVA', p_a])
        tests_style.append(['', 'fc_pink' if p_a < ALPHA else ''])
    
        # Kruskal-Wallis H-test for independent samples
        s_kw, p_kw = stats.kruskal(*gg)
        tests.append(['Kruskal-Wallis', p_kw])
        tests_style.append(['', 'fc_pink' if p_kw < ALPHA else ''])

        if parametric:
            res['test'], res['p'] = 'ANOVA', p_a
        else:
            res['test'], res['p'] = 'Kruskal-Wallis', p_kw
            
    if summary:
        for g, g_name in zip(gg, g_names):
            if parametric:
                res[g_name] = format_float(np.mean(g)) + ' ±' + format_float(np.std(g, ddof=1))
            else:
                p25, p50, p75 = np.percentile(g, [25, 50, 75], interpolation='midpoint')
                res[g_name] = f'{format_float(p50)} ({format_float(p25)}, {format_float(p75)})'

    if plot:
        table_0 = [
            [None] + list(g_names) + ['total'],
            ['n'] + [len(g) for g in gg] + [len(var_nona)],
            ['missing'] + list(g_missing) + [na_loc.sum()],
            ['median'] + [np.median(g) for g in gg] +[np.median(var_nona)],
            ['mean'] + [np.mean(g) for g in gg] + [np.mean(var_nona)],
            ['SD'] + [np.std(g, ddof=1) for g in gg] + [np.std(var_nona, ddof=1)],
        ]
        style_0 = [[None] * len(table_0[0])] * len(table_0)
        style_0[0] = [None] + [f'fc_C{x}' for x in range(n_groups)] + [None]
        style_0[4] = [None] + ['fc_lightgreen' if x else '' for x in possibly_normal] + [None]
        hist_rows = n_groups if scale == 'continuous' else 1
        fig, ax = _make_fig(res, table_0, style_0, rows=hist_rows)
        
        if scale == 'continuous':
            _plot_histograms(var_nona.values.flatten(), gg, possibly_normal, possibly_lognormal, ax)
        elif scale == 'categorical':
            counts = var_nona.groupby([var_nona,grp_nona]).count().unstack().T
            _plot_bars(counts, ax[1])
        else:
            raise Exception(f'unknown scale: {scale}')
        
        for x in range(len(g_names)):
            sm.qqplot(gg[x], ax=ax[2], markerfacecolor='none', markeredgecolor=f'C{x}', line='s')
        ax[2].set_title('Q-Q normal~sample')
        ax[2].get_xaxis().label.set_visible(False)
        ax[2].get_yaxis().label.set_visible(False)
        
        table = plot_table(tests, style=tests_style, ax=ax[3])
        table.auto_set_font_size(False)
        table.set_fontsize(FONTSIZE)
        plt.show()
    return res













def paired_difference_test(measurements, plot=True, scale=None, parametric=None, **kwa):
    formula = f'{measurements.columns[0]} vs. {measurements.columns[1]}'
    res = {'formula': formula, 'scale': scale, 'test': None, 'p': np.nan}

    mm_nona = measurements.dropna()
    mm_nona_list = [s for name, s in mm_nona.items()]
    n_mm = len(mm_nona_list)
        
    tests = [['test', 'p-value']]
    tests_style = [['bold center'] * 2]
    
    # Shapiro-Wilk test for normal distribution
    p_shapiro, p_logshapiro = [], []
    for name, s in mm_nona.items():
        if len(s) < 3 or np.ptp(s) == 0:
            p, lp = 0, 0
        else:
            _, p = stats.shapiro(s)
            _, lp = stats.shapiro(np.log(s)) if s.min() > 0 else (0, 0)
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
        
    if distribution == 'lognormal':
        warnings.warn(f'{list(measurements.columns)}: all measurements possibly'
                      'lognormal. Tests not implemented, yet!')
    
    if measurements.shape[1] == 1:
        raise NotImplementedError('Just one measurement.') # this should never happen
    elif measurements.shape[1] == 2:
        # t-test for the means of paired samples
        s_t, p_t = stats.ttest_rel(*mm_nona_list)
        tests.append(['paired t', p_t])
        tests_style.append(['', 'fc_pink' if p_t < ALPHA else ''])
        
        # Wilcoxon signed-rank test
        s_rs, p_rs = stats.wilcoxon(*mm_nona_list, alternative='two-sided')
        tests.append(['signed-rank', p_rs])
        tests_style.append(['', 'fc_pink' if p_rs < ALPHA else ''])
        if parametric is None:
            parametric = distribution == 'normal'
        
        if parametric:
            res['test'], res['p'] = 'paired t', p_t
        else:
            res['test'], res['p'] = 'signed-rank', p_rs
    else:
        res['test'], res['p'] = 'ANOVA', None
        warnings.warn(f'ANOVA not implemented, yet!')
        

    if plot:
        table = [
            [None] + list(measurements.columns),
            ['n'] + [len(s) for s in mm_nona_list],
            ['missing'] + [measurements.shape[0] - mm_nona.shape[0]] * n_mm,
            ['median'] + [np.median(s) for s in mm_nona_list],
            ['mean'] + [np.mean(s) for s in mm_nona_list],
            ['SD'] + [np.std(s, ddof=1) for s in mm_nona_list],
        ]
        style = [[None] * len(table[0])] * len(table)
        style[0] = [None] + [f'fc_C{x}' for x in range(measurements.shape[1])]
        style[4] = [None] + ['fc_lightgreen' if x else '' for x in possibly_normal]
        
        hist_rows = measurements.shape[1] if scale == 'continuous' else 1
        fig, ax = _make_fig(res, table, style, rows=hist_rows)
        
        if scale == 'continuous':
            _plot_histograms(mm_nona.values.flatten(), mm_nona_list,
                             possibly_normal, possibly_lognormal, ax)
        elif scale == 'categorical':
            print(mm_nona.shape)
            counts = (mm_nona.melt().assign(count=1).groupby(['variable', 'value'])
                      .count().unstack('variable').fillna(0))
            _plot_bars(counts.T, ax[1])
        else:
            raise Exception(f'unknown scale: {scale}')
        
        for x, s in enumerate(mm_nona_list):
            sm.qqplot(s, ax=ax[2], markerfacecolor='none', markeredgecolor=f'C{x}', line='s')
        ax[2].set_title('Q-Q normal~sample')
        ax[2].get_xaxis().label.set_visible(False)
        ax[2].get_yaxis().label.set_visible(False)
        
        table = plot_table(tests, style=tests_style, ax=ax[3])
        table.auto_set_font_size(False)
        table.set_fontsize(FONTSIZE)
        plt.show()
        
#         plt.subplots_adjust(hspace=0.05, wspace=0.05)
#         fig.tight_layout()
    return res



def independent_proportion_test(var, grouping, plot=True, summary=False, scale=None, **kwa):
    res = {'formula': f'{var.name} ~ {grouping.name}', 'scale': scale, 'test': None, 'p': np.nan}
    na_loc, var_nona, grp_nona, g_names, gg, g_missing = _split_to_groups(var, grouping)
    n_groups = len(gg)
    
    tests = [['test', 'p-value']]
    tests_style = [['bold center'] * 2]
    counts = pd.crosstab(var_nona, grp_nona)
    chi2_valid = '' if counts.values.min() >= 5 else 'fc_pink'

    if counts.shape[0] > 1:
        test = 'Pearson chi^2'
        chi2, p, dof, exp = stats.chi2_contingency(counts, correction=False)
#         res['test'], res['p'] = test, p
        tests.append([r'$\chi^2$ Pearson', p])
        tests_style.append([chi2_valid, 'fc_pink' if p < ALPHA else ''])

        test = 'Yates chi^2'
        chi2, p, dof, exp = stats.chi2_contingency(counts, correction=True)
        res['test'], res['p'] = test, p
        tests.append([r'$\chi^2$ Yates', p])
        tests_style.append([chi2_valid, 'fc_pink' if p < ALPHA else ''])

        if counts.shape[1] < 2:
            pass
        elif counts.shape == (2, 2): 
            test = 'Fisher exact'
            oddsratio, p = stats.fisher_exact(counts, alternative='two-sided')
            res['test'], res['p'] = test, p
            tests.append([test, p])
            tests_style.append(['', 'fc_pink' if p < ALPHA else ''])
            tests.append(['odds ratio', oddsratio])
            tests_style.append([None, None])
        elif startr():
            test = 'Fisher exact'
            p = r_stats.fisher_test(counts.values)[0][0]
            res['test'], res['p'] = test, p
            tests.append([test, p])
            tests_style.append(['', 'fc_pink' if p < ALPHA else ''])
    else:
        # There is just single value. Let's add complementary binary value.
        val = counts.iloc[0].name
        complementary = {'0':1, '1':0, 'True':False, 'False':True}.get(str(val))
        if complementary is not None:
            tcounts = counts.T
            tcounts[complementary] = 0
            counts = tcounts.T.sort_index()
    if summary:
        for g_name, count in counts.items():
            res[str(g_name)] = f'{count.iloc[-1]}/{count.sum()} ({count.iloc[-1] / count.sum() * 100:2.0f}%)'
    if plot:
        table_0 = [[''] + list(g_names) + ['total']]
        style_0 = [[None] + [f'fc_C{x}' for x in range(counts.shape[1])] + ['normal']]
        sums = counts.sum()
        total = sums.sum()
        warn = lambda x: 'fc_pink' if x < 5 else ''
        for val, row in counts.iterrows():
            table_0.append([val] + [_perc(x, col_total) for x, col_total in zip(row, sums)] + [_perc(row.sum(), total)])
            style_0.append([''] + ['right ' + warn(x) for x in row] + ['right'])
        table_0.append(['total'] + [_perc(x, total) for x in sums] + [_perc(total, total)])
        style_0.append([''] + ['right' for x in sums] + ['right'])
        table_0.append(['missing'] + [x for x in g_missing] + [sum(g_missing)])
        style_0.append([''] + ['right' for x in g_missing] + ['right'])
        fig, ax = _make_fig(res, table_0, style_0)

        _plot_bars(counts.T, ax[1])

        ax[2].set_title('Observed vs Expected')
        sums = counts.sum(axis=1)
        if sums.shape[0] > 1:
            ax[2].plot([0, sums.iloc[0]], [0, sums.iloc[1]], color='black')
            for ii, col in counts.T.iterrows():
                ax[2].plot(col.iloc[0], col.iloc[1], 'o')
            ax[2].set_aspect('equal', adjustable='box')
            ax[2].set_xlabel(sums.index[0])
            ax[2].set_ylabel(sums.index[1])

        table = plot_table(tests, style=tests_style, ax=ax[3])
        table.auto_set_font_size(False)
        table.set_fontsize(FONTSIZE)
        plt.show()
    return res

def paired_proportion_test(measurements, plot=True, summary=False, scale=None, **kwa):
    formula = f'{measurements.columns[0]} vs. {measurements.columns[1]}'
    res = {'formula': formula, 'scale': scale, 'test': None, 'p': np.nan}

    mm_nona = measurements.dropna()
    mm_nona_list = [s for name, s in mm_nona.items()]
    n_mm = len(mm_nona_list)
    counts = pd.crosstab(*mm_nona_list)
        
    tests = [['test', 'p-value']]
    tests_style = [['bold center'] * 2]
    
    if counts.shape[0] == 1:
        # There is just single value. Let's add complementary binary value.
        val = counts.iloc[0].name
        complementary = {'0':1, '1':0, 'True':False, 'False':True}.get(str(val))
        if complementary is not None:
            tcounts = counts.T
            tcounts[complementary] = 0
            counts = tcounts.T.sort_index()
    elif counts.shape == (2,2):
        test = 'McNemar'
        mcnemar = sm.stats.mcnemar(counts, exact=False, correction=True)
        tests.append([test, mcnemar.pvalue])
        mcnemar_valid = '' if (counts.iloc[0, 1] + counts.iloc[0, 1]) >= 10 else 'fc_pink'
        tests_style.append([mcnemar_valid, 'fc_pink' if mcnemar.pvalue < ALPHA else ''])
    else:
        pass

    return tests



    if summary:
        for g_name, count in counts.items():
            res[str(g_name)] = f'{count.iloc[-1]} ({count.iloc[-1] / count.sum() * 100:2.0f}%)'
    if plot:
        table_0 = [[''] + list(g_names) + ['total']]
        style_0 = [[None] + [f'fc_C{x}' for x in range(counts.shape[1])] + ['normal']]
        sums = counts.sum()
        total = sums.sum()
        warn = lambda x: 'fc_pink' if x < 5 else ''
        for val, row in counts.iterrows():
            table_0.append([val] + [_perc(x, col_total) for x, col_total in zip(row, sums)] + [_perc(row.sum(), total)])
            style_0.append([''] + ['right ' + warn(x) for x in row] + ['right'])
        table_0.append(['total'] + [_perc(x, total) for x in sums] + [_perc(total, total)])
        style_0.append([''] + ['right' for x in sums] + ['right'])
        table_0.append(['missing'] + [x for x in g_missing] + [sum(g_missing)])
        style_0.append([''] + ['right' for x in g_missing] + ['right'])
        fig, ax = _make_fig(res, table_0, style_0)

        _plot_bars(counts.T, ax[1])

        ax[2].set_title('Observed vs Expected')
        sums = counts.sum(axis=1)
        if sums.shape[0] > 1:
            ax[2].plot([0, sums.iloc[0]], [0, sums.iloc[1]], color='black')
            for ii, col in counts.T.iterrows():
                ax[2].plot(col.iloc[0], col.iloc[1], 'o')
            ax[2].set_aspect('equal', adjustable='box')
            ax[2].set_xlabel(sums.index[0])
            ax[2].set_ylabel(sums.index[1])

        table = plot_table(tests, style=tests_style, ax=ax[3])
        table.auto_set_font_size(False)
        table.set_fontsize(FONTSIZE)
        plt.show()
    return res

def _perc(x, total):
    return f'{x} ({x / total * 100:2.0f}%)'
    

r_stats = None

def startr():
    global r_stats
    if r_stats is None:
        try:
            import rpy2
            from rpy2.robjects import numpy2ri
            numpy2ri.activate()
            from rpy2.robjects.packages import importr
            r_stats = importr('stats')
        except ModuleNotFoundError:
            warnings.warn(f'Fisher test of table larger than 2x2 requires R and rpy2 installed.')
    return r_stats


def _make_fig(res, table, style, rows=1):
    fig = plt.figure(figsize=(14, 2), constrained_layout=False, dpi=75)
    spec = fig.add_gridspec(rows, 4, width_ratios=(4,2,2,2), hspace=.2)
    ax = 4 * [None]
    ax[0] = fig.add_subplot(spec[:,0])
    ax[1] = [fig.add_subplot(spec[0,1])]
    ax[1] += [fig.add_subplot(spec[row,1], sharex=ax[1][0], sharey=ax[1][0]) 
              for row in range(1, rows)]
    ax[2] = fig.add_subplot(spec[:,2])
    ax[3] = fig.add_subplot(spec[:,3])
    
    ax[0].set_title(res['formula'], loc='left')
    ax[1][0].set(title=res['scale'])
    table_artist = plot_table(table, style=style, loc='full', ax=ax[0])
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(FONTSIZE)
    
    return fig, ax
    
def _plot_histograms(all_values, groups_list, possibly_normal, possibly_lognormal, ax):
    n_mm = len(groups_list)
    nbins = max([10] + [len(g) // 10 for g in groups_list])
    X = np.linspace(all_values.min(), all_values.max(), 100)
    _, bins = np.histogram(all_values, nbins)
    for ii, s in enumerate(groups_list):
        ax[1][ii].hist(groups_list[ii], bins=bins, color=f'C{ii}')
        if ii < (n_mm - 1):
#                     ax[1][ii].tick_params(bottom=False, labelbottom=False)
            ax[1][ii].get_xaxis().set_visible(False)
#                     ax[1][ii].set_xticks([])
        binwidth = bins[1] - bins[0]
        normal, lognormal = possibly_normal[ii], possibly_lognormal[ii]
        if normal:
            Y = stats.norm.pdf(X, loc=s.mean(), scale=s.std(ddof=1)) * len(s) * binwidth
            ax[1][ii].plot(X, Y, color='k')#f'C{ii}'

def _plot_bars(counts, ax):
    
    ax = ax[0]
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
    
def _split_to_groups(var, grouping):
    grouping_na = grouping.isna()
    grouping_na_sum = grouping_na.sum()
    if grouping_na_sum > 0:
        var = var[~ grouping_na]
        grouping = grouping[~ grouping_na]
        warnings.warn(f'{grouping_na_sum} rows removed because of missing values in grouping variable.')
    na_loc = var.isna()
    var_nona = var[~ na_loc]
    grp_nona = grouping[~ na_loc]
    if isinstance(grp_nona, pd.Series) and grp_nona.dtype.name == 'category':
        g_names = grp_nona.cat.categories
        gg = [var_nona[grp_nona == name] for name in g_names]
        g_missing = [na_loc[grouping == name].sum() for name in g_names]
    else:
        glu = np.sort(grp_nona.unique())
        g_names = [str(g) for g in glu]
        gg = [var_nona[grp_nona == g] for g in glu]
        g_missing = [na_loc[grouping == g].sum() for g in glu]
    return na_loc, var_nona, grp_nona, g_names, gg, g_missing

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

