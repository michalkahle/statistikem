import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import lifelines
from lifelines.utils import median_survival_times
import matplotlib

def kmplot(durations, 
           grouping, 
           event, 
           data=None, 
           counts=True,
           ymin=None, 
           ax=None, 
           plot='survival',
           ploc='lower left', **kwargs):
    '''
    Plot Kaplan-Meier curves for survival analysis.

    Parameters:
    durations (str): Column name in `data` that represents the durations.
    grouping (str): Column name in `data` that represents the grouping variable.
    event (str, optional): Column name in `data` that represents the event of interest. Defaults to 'Amputation'.
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
    res = {}
    if ax is None:
        fig, ax = plt.subplots()
    group_names =  data[grouping].cat.categories if data[grouping].dtype == 'category' else data[grouping].dropna().unique()
    for g_name in group_names:
        group = data[data[grouping] == g_name]
        
    
        kmf = lifelines.KaplanMeierFitter()
        kmf.fit(group[durations], group[event], label=str(g_name))
        ci = median_survival_times(kmf.confidence_interval_).values[0]
        res[g_name] = f"{kmf.median_survival_time_:.2f} CI95=({ci[0]:.2f}, {ci[1]:.2f})" 
        if plot == 'survival':
            kmf.plot(show_censors=True, 
                      censor_styles=dict(marker='|', alpha=.3), 
                      # loc=kwargs.get('xloc'),
                      ax=ax)
        else:
            kmf.plot_cumulative_density(show_censors=True, 
                      censor_styles=dict(marker='|', alpha=.3), 
                      # loc=kwargs.get('xloc'),
                      ax=ax)
        if counts:
            models.append(kmf)
        for contrast, contrast_g_name in previous:
            p = lifelines.statistics.logrank_test(
                contrast[durations], group[durations], contrast[event], group[event]).p_value
            p_label = 'p=' if len(group_names) < 3 else f'{contrast_g_name} vs {g_name}: p='
            pp.append(p_label + format_p(p))
        previous.append((group, g_name))
        
    lifelines.plotting.add_at_risk_counts(*models, rows_to_show=['At risk'],  ax=ax) #, 'Censored', 'Events'
    ax.set_title(f'{grouping}')
    # ax.set_ylabel('Survival')
    if len(pp) > 0:
        at = matplotlib.offsetbox.AnchoredText('\n'.join(pp), loc=ploc, frameon=False)
        ax.add_artist(at)
    ax.set_ylim([ymin, 1.0])
    # steps = {0.0:5, 0.5:6}[ymin]
    # y = np.linspace(ymin, 1, steps)
    # ax.set_yticks(y)
    # ax.set_yticklabels((y * 100).astype(int))
    ax.set_xlabel('Years Since ACT')
    # ax.legend(loc='lower left')
    [ch.set_edgecolor(None) for ch in ax.get_children() if isinstance(ch, matplotlib.collections.PolyCollection)]
    return res


