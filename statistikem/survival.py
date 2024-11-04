import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import lifelines
import matplotlib
from statistikem.helpers import format_p
from statistikem.helpers import _get_series, guess_scale
import warnings

def cox(predictors, data, duration_col, event_col, cluster_col=None, check_ph=False):
    if type(predictors) == str:
        predictors = [predictors]
    fig, ax = plt.subplots(1,1, figsize=(6, .5 + .5*len(predictors)))
    cph = lifelines.CoxPHFitter()
    cols = [duration_col, event_col] + predictors
    if cluster_col is not None:
        cols += [cluster_col]
    data = data[cols].dropna()
    cph.fit(data, duration_col=duration_col, event_col=event_col, cluster_col=cluster_col)
    cph.plot(hazard_ratios=True, ax=ax)
    maxlen = max([len(x) for x in predictors])
    res = [f'{r.name: <{maxlen}} HR={r["exp(coef)"]:.3f} ({r["exp(coef) lower 95%"]:.3f}, {r["exp(coef) upper 95%"]:.3f}), p={r["p"]:.3f}' 
          for name, r in cph.summary.iterrows()]
    ax.set_ylim(-0.5, len(predictors)-0.5)
    if check_ph:
        cph.check_assumptions(data, p_value_threshold=0.05, advice=True, show_plots=True)
    return '\n'.join(res)

def univariate_cox(predictors, data, duration_col, event_col, cluster_col=None, **kwargs):
    if type(predictors) == str:
        predictors = [predictors]
    n = len(predictors)
    fig, ax = plt.subplots(n, 1, figsize=(6, .5 + 1.0*n), squeeze=False, sharex=kwargs.pop('sharex', None))
    
    res = []
    for n, var in enumerate(predictors):
        cph = lifelines.CoxPHFitter()
        cols = [duration_col, event_col, var]
        if cluster_col is not None:
            cols += [cluster_col]
        cph.fit(data[cols].dropna(), duration_col=duration_col, event_col=event_col, cluster_col=cluster_col)
        cph.plot(hazard_ratios=True, ax=ax[n,0])
        r = cph.summary
        res.append({
            'predictor': var,
            'HR': r.loc[var, 'exp(coef)'], 
            '95%CI_low': r.loc[var, "exp(coef) lower 95%"], 
            '95%CI_high': r.loc[var, "exp(coef) upper 95%"],
            'p': r.loc[var, "p"]
        })
    fig.tight_layout()
    return pd.DataFrame(res)


def survplot(durations, 
           event, 
           grouping=None, 
           data=None,
           event_of_interest=None,
           counts=True,
           ylim=(0, 1), 
           ax=None, 
           xlabel=None,
           direction='survival',
           estimator='KM',
           ploc=None,
           show_censors=True,
           tests=True,
           title=None,
           output=None,
           **kwargs):
    '''
    Plot Kaplan-Meier curves for survival analysis.

    Parameters:
    durations (str|Series): Series or column name in `data` that represents the durations.
    event (str|Series): Series or column name in `data` that represents the events.
    event_of_interest (optional): value in `event` that represents the event of interest. 
    grouping (str|Series, optional): Series or column name in `data` that represents the grouping variable.
    data (DataFrame, optional): Pandas DataFrame that contains the data.
    counts (bool, optional): If True, add at risk counts at the bottom of the plot. Defaults to True.
    ylim (tuple of float, optional): The limits for the y-axis. Defaults to (0, 1). Can be replaced with `None`.
    ax (matplotlib.axes.Axes, optional): The axes upon which to draw the plot. If None, the plot is drawn on a new set of axes.
    direction (str, optional): Type of plot to draw. 'survival' for survival function, 'density' for cumulative density function. Defaults to 'survival'.
    ploc (str, optional): Location of the p-value in the plot. Defaults to 'lower left'.
    **kwargs: Arbitrary keyword arguments.

    Returns:
    dict: A dictionary where keys are group names and values are median survival times with 95% confidence intervals.
    '''
    table_times =  kwargs.pop('table_times', [1, 3, 5, 10, 15, 20, 25])
    durations = _get_series(durations, data)
    event = _get_series(event, data)
    grouping = _get_series(grouping, data)
    models = []
    previous = []
    pp = []
    res = []
    scale = guess_scale(event)
    if estimator == 'KM' and scale != 'binary':
        if scale == 'categorical' and event_of_interest is not None:
            event = event == event_of_interest if type(event_of_interest) == str else event.isin(event_of_interest)
        else:
            raise TypeError(f'Event should be a binary variable. Instead, {scale} variable was provided.')
    if estimator == 'AJ':
        if scale != 'categorical':
            warnings.warn(f'Event should be a categorical variable. Instead, {scale} variable was provided.')
        if event_of_interest is None:
            raise ValueError('The parameter `event_of_inrerest` must be set.')
        if type(event) != int:
            event, index = pd.factorize(event)
            event = pd.Series(event + 1, index=durations.index)
            event_of_interest = index.get_loc(event_of_interest) + 1




    if ploc is None:
        ploc = 'lower left' if direction == 'survival' else 'lower right'
    if ax is None:
        fig, ax = plt.subplots(dpi=75)
    if grouping is None:
        grouping = pd.Series(len(event) * ['All'])
    group_names =  grouping.cat.categories if grouping.dtype == 'category' else np.sort(grouping.dropna().unique())
    for g_name in group_names:
        group_index = (grouping == g_name).values
        g_durations = durations[group_index]
        g_event = event[group_index]
    
        if estimator == 'KM':
            model = lifelines.KaplanMeierFitter()
            kwargs['censor_styles'] = dict(marker='|', alpha=.3)
            kwargs['show_censors'] = show_censors
            model.fit(g_durations, g_event, label=str(g_name))
            survival_function_ = model.survival_function_
        elif estimator == 'AJ':
            model = lifelines.AalenJohansenFitter()
            with warnings.catch_warnings(action='ignore'):
                model.fit(g_durations, g_event, event_of_interest=event_of_interest, label=str(g_name))
            survival_function_ = model.cumulative_density_

        if output in ('table', 'table_ci'):
            if estimator == 'KM':
                ll = list(model.survival_function_at_times(table_times))
            elif estimator == 'AJ':
                # ll = model.cumulative_density_at_times(table_times) # not implemented
                ll = step_function_at_times(model.cumulative_density_.iloc[:,0], table_times)
            if output == 'table_ci':
                ci_l = step_function_at_times(model.confidence_interval_.iloc[:,0], table_times)
                ci_u = step_function_at_times(model.confidence_interval_.iloc[:,1], table_times)

                row = [f'{ll[ii]:.2f} ({ci_l[ii]:.2f}, {ci_u[ii]:.2f})' for ii in range(len(table_times))]
            else:
                row = [f'{x:.2f}' for x in ll]
            res.append([grouping.name, g_name] + row)
        elif output == 'median_survival':
            quantile = kwargs.get('quantile', 0.5)
            q = lifelines.utils.qth_survival_time(quantile, survival_function_)
            ci = lifelines.utils.qth_survival_times(quantile, model.confidence_interval_).values[0]
            res.append([grouping.name, g_name, f"{q:.2f} CI95=({ci[0]:.2f}, {ci[1]:.2f})"])

        if direction == 'survival':
            model.plot(ax=ax, **kwargs)
        else:
            model.plot_cumulative_density(ax=ax, **kwargs)
        if counts:
            models.append(model)
        if tests:
            g_event_bin = g_event if estimator == 'KM' else g_event == event_of_interest
            for contrast_durations, contrast_event, contrast_name in previous:
                p = lifelines.statistics.logrank_test(
                    contrast_durations, g_durations, contrast_event, g_event_bin).p_value
                p_label = 'p=' if len(group_names) < 3 else f'{contrast_name} vs {g_name}: p='
                pp.append(p_label + format_p(p))
                if output == 'tests':
                    res.append([grouping.name, p_label + format_p(p)])
            previous.append((g_durations, g_event_bin, g_name))
        
    lifelines.plotting.add_at_risk_counts(*models, rows_to_show=['At risk'],  ax=ax) #, 'Censored', 'Events'
    ax.set_title(title if title is not None else grouping.name)
    if len(pp) > 0:
        at = matplotlib.offsetbox.AnchoredText('\n'.join(pp), loc=ploc, frameon=False)
        ax.add_artist(at)
    ax.set_ylim(ylim)
    ax.set_xlim(left=0)

    # y = np.linspace(0, 1, 5)
    # ax.set_yticks(y)
    # ax.set_yticklabels((y * 100).astype(int))
    ax.set_xlabel(xlabel if xlabel is not None else durations.name)
    # ax.legend(loc='lower left')
    [ch.set_edgecolor(None) for ch in ax.get_children() if isinstance(ch, matplotlib.collections.PolyCollection)]
    plt.tight_layout()
    if output is not None:
        res = pd.DataFrame(res)
        if output in ('table', 'table_ci'):
            res.columns = ['variable', 'group'] + table_times
        elif output == 'median_survival':
            res.columns = ['variable', 'group', 'median survival']
        elif output == 'tests':
            res.columns = ['variable', 'tests']
        return res

def survplots(duration, event, cols, data, xlabel=None, plot='survival', output=None, **kwargs):
    n = len(cols)
    r = n//4 + (0 if n%4==0 else 1)
    fig, axs = plt.subplots(r, 4, squeeze=True, figsize=(20, 5*r))
    axs = axs.flatten()
    res = [survplot(duration, event, col, data=data, ax=axs[ii], direction=plot, xlabel=xlabel, output=output, **kwargs) for ii, col in enumerate(cols)]
    fig.tight_layout()
    if output is not None:
        return pd.concat(res)

def step_function_at_times(s, timeline):
    return [s.iloc[np.searchsorted(s.index, time, side='left') - 1] for time in timeline]