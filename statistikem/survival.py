import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import lifelines
import matplotlib
from statistikem.helpers import format_p

def kmplot(durations, 
           event, 
           grouping=None, 
           data=None, 
           counts=True,
           ymin=None, 
           ax=None, 
           xlabel=None,
           plot='survival',
           ploc=None,
           show_censors=True,
           tests=True,
           title=None,
           quantile=0.5, # median
           **kwargs):
    '''
    Plot Kaplan-Meier curves for survival analysis.

    Parameters:
    durations (str): Column name in `data` that represents the durations.
    event (str): Column name in `data` that represents the event of interest.
    grouping (str, optional): Column name in `data` that represents the grouping variable.
    data (DataFrame, optional): Pandas DataFrame that contains the data. Defaults to None.
    counts (bool, optional): If True, add at risk counts at the bottom of the plot. Defaults to True.
    ymin (float, optional): The lower limit for the y-axis. Defaults to None.
    ax (matplotlib.axes.Axes, optional): The axes upon which to draw the plot. If None, the plot is drawn on a new set of axes. Defaults to None.
    plot (str, optional): Type of plot to draw. 'survival' for survival function, 'density' for cumulative density function. Defaults to 'survival'.
    ploc (str, optional): Location of the p-value in the plot. Defaults to 'lower left'.
    **kwargs: Arbitrary keyword arguments.

    Returns:
    dict: A dictionary where keys are group names and values are median survival times with 95% confidence intervals.
'''
    models = []
    previous = []
    pp = []
    res = []
    if ploc is None:
        ploc = 'lower left' if plot == 'survival' else 'lower right'
    if ax is None:
        fig, ax = plt.subplots(dpi=75)
    if grouping is None:
        grouping = 'All'
        data = data.copy()
        data[grouping] = 'All'
    group_names =  data[grouping].cat.categories if data[grouping].dtype == 'category' else np.sort(data[grouping].dropna().unique())
    for g_name in group_names:
        group = data[data[grouping] == g_name]
        
    
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(group[durations], group[event], label=str(g_name))
        q = lifelines.utils.qth_survival_time(quantile, kmf.survival_function_)
        ci = lifelines.utils.qth_survival_times(quantile, kmf.confidence_interval_).values[0]
        res.append([grouping, g_name, f"{q:.2f} CI95=({ci[0]:.2f}, {ci[1]:.2f})"])
        if plot == 'survival':
            kmf.plot(show_censors=show_censors, 
                      censor_styles=dict(marker='|', alpha=.3), 
                       ax=ax, **kwargs)
        else:
            kmf.plot_cumulative_density(show_censors=show_censors, 
                      censor_styles=dict(marker='|', alpha=.3), 
                      ax=ax, **kwargs)
        if counts:
            models.append(kmf)
        if tests:
            for contrast, contrast_g_name in previous:
                p = lifelines.statistics.logrank_test(
                    contrast[durations], group[durations], contrast[event], group[event]).p_value
                p_label = 'p=' if len(group_names) < 3 else f'{contrast_g_name} vs {g_name}: p='
                pp.append(p_label + format_p(p))
        previous.append((group, g_name))
        
    lifelines.plotting.add_at_risk_counts(*models, rows_to_show=['At risk'],  ax=ax) #, 'Censored', 'Events'
    ax.set_title(title if title is not None else grouping)
    # ax.set_ylabel('Survival')
    if len(pp) > 0:
        at = matplotlib.offsetbox.AnchoredText('\n'.join(pp), loc=ploc, frameon=False)
        ax.add_artist(at)
    if ymin is not None:
        ax.set_ylim([ymin, 1.0])

    # steps = {0.0:5, 0.5:6}[ymin]
    # y = np.linspace(ymin, 1, steps)
    # ax.set_yticks(y)
    # ax.set_yticklabels((y * 100).astype(int))
    ax.set_xlabel(xlabel if xlabel is not None else durations)
    # ax.legend(loc='lower left')
    [ch.set_edgecolor(None) for ch in ax.get_children() if isinstance(ch, matplotlib.collections.PolyCollection)]
    plt.tight_layout()
    return res

def kmplots(duration, event, cols, data, xlabel=None, plot='survival', **kwargs):
    res = []
    n = len(cols)
    r = n//4 + (0 if n%4==0 else 1)
    fig, axs = plt.subplots(r, 4, squeeze=True, figsize=(20, 5*r))
    axs = axs.flatten()
    for ii, col in enumerate(cols):
        res += kmplot(duration, event, col, data=data, ax=axs[ii], plot=plot, xlabel=xlabel, **kwargs)
    fig.tight_layout()
    q_label = 'median survival time' if kwargs.get('quantile') is None else f'{kwargs.get("quantile")*100:.0f} percentile survival time'
    return pd.DataFrame(res, columns=['factor', 'group', q_label])

def kmtable(durations, 
            event, 
            grouping=None,
            data=None, 
            times=[1, 3, 5, 10, 15, 20, 25], 
            **kwargs):
    table = {}
    if grouping is None:
        grouping = 'All'
        data = data.copy()
        data[grouping] = 'All'
    group_names =  data[grouping].cat.categories if data[grouping].dtype == 'category' else np.sort(data[grouping].dropna().unique())
    for g_name in group_names:
        group = data[data[grouping] == g_name]
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(group[durations], group[event], label=str(g_name), timeline=times)
        s = kmf.survival_function_.iloc[:,0]
        ci_l = kmf.confidence_interval_.iloc[:,0]
        ci_u = kmf.confidence_interval_.iloc[:,1]
        ll = [f'{proportion*100:.0f}%' for proportion in s]
        # ll = [f'{s[time]:.2f}, 95% CI [{ci_l[time]:.2f}, {ci_u[time]:.2f}]' for time in s.index]
        table[g_name] = ll
    table = pd.DataFrame(table, index=s.index)
    return table