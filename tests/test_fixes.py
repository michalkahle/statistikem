import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statistikem as si
from statistikem import helpers, survival


@pytest.fixture
def paired_long():
    rng = np.random.default_rng(0)
    n = 30
    base = rng.normal(10, 2, n)
    return pd.DataFrame({
        'id': np.tile(np.arange(n), 3),
        'visit': np.repeat(['a', 'b', 'c'], n),
        'x': np.concatenate([base, base + rng.normal(1, 1, n), base + rng.normal(2, 1, n)]),
    })


@pytest.fixture
def surv_df():
    rng = np.random.default_rng(1)
    n = 80
    return pd.DataFrame({
        't': rng.exponential(5, n),
        'e': rng.random(n) < 0.6,
        'g': np.repeat(['a', 'b'], n // 2),
        'x': rng.normal(size=n),
    })


def test_paired_three_timepoints_get_a_test(paired_long):
    res = si.compare_one('x', 'visit', 'id', data=paired_long, plot=False)
    assert res['test'] in ('RM ANOVA', 'Friedman')
    assert res['p'] < 0.05
    assert si.compare_one('x', 'visit', 'id', data=paired_long, plot=False, parametric=False)['test'] == 'Friedman'


def test_paired_normality_is_judged_on_differences():
    rng = np.random.default_rng(2)
    base = rng.lognormal(3, 1, 60)
    df = pd.DataFrame({'id': np.tile(np.arange(60), 2), 'visit': np.repeat(['a', 'b'], 60),
                       'x': np.concatenate([base, base + rng.normal(1, 1, 60)])})
    assert si.compare_one('x', 'visit', 'id', data=df, plot=False)['test'] == 'paired t'


def test_paired_identical_values_do_not_raise():
    df = pd.DataFrame({'id': np.tile(np.arange(10), 2), 'visit': np.repeat(['a', 'b'], 10),
                       'x': np.tile(np.arange(10.0), 2)})
    res = si.compare_one('x', 'visit', 'id', data=df, plot=False, parametric=False)
    assert np.isnan(res['p'])


def test_two_groups_use_welch():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({'g': np.repeat(['a', 'b'], 40), 'x': np.concatenate([rng.normal(0, 1, 40), rng.normal(1, 1, 40)])})
    res = si.compare_one('x', 'g', data=df, plot=False, parametric=True)
    assert res['test'] == 'Welch t'
    assert res['p'] == pytest.approx(stats.ttest_ind(df.query("g == 'a'")['x'], df.query("g == 'b'")['x'], equal_var=False).pvalue)


def test_p_corr_skips_untested_predictors():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({'g': np.repeat(['a', 'b'], 20), 'x': rng.normal(size=40), 'const': 1})
    out = si.compare(['x', 'const'], 'g', data=df, plot=False)
    assert np.isnan(out.loc[out['predictor'] == 'const', 'p_corr']).all()
    assert out.loc[out['predictor'] == 'x', 'p_corr'].notna().all()


def test_misspelled_argument_raises():
    df = pd.DataFrame({'g': np.repeat(['a', 'b'], 20), 'x': np.arange(40.0)})
    with pytest.raises(TypeError):
        si.compare('x', 'g', data=df, plot=False, parametic=True)


def test_few_distinct_numeric_values_warn():
    df = pd.DataFrame({'g': np.repeat(['a', 'b'], 20), 'x': np.tile([1, 2, 3, 4], 10)})
    with pytest.warns(UserWarning, match='scale='):
        si.compare_one('x', 'g', data=df, plot=False)


def test_correlate_rejects_unknown_kind():
    df = pd.DataFrame({'a': np.arange(10.0), 'b': np.arange(10.0) ** 2})
    with pytest.raises(ValueError):
        si.correlate(df, kind='pearson', plot=False)


def test_bootstrap_is_reproducible_and_centred_on_sample_mean():
    s = pd.Series(np.random.default_rng(5).normal(size=50))
    a, b = helpers.ci_mean_bootstrap(s, as_df=False), helpers.ci_mean_bootstrap(s, as_df=False)
    assert a == b
    assert a[0] == s.mean()


def test_ci_mean_normal_matches_scipy():
    s = pd.Series(np.random.default_rng(6).normal(size=12))
    lo, hi = stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=s.sem())
    assert helpers.ci_mean_normal(s)[1:] == pytest.approx((lo, hi))


def test_step_function_at_times():
    s = pd.Series([0.0, 0.2, 0.5], index=[0.0, 1.0, 3.0])
    assert survival.step_function_at_times(s, [-1, 0.5, 1, 2, 3, 10]) == [0.0, 0.0, 0.2, 0.2, 0.5, 0.5]


def test_univariate_cox_reports_n_and_events(surv_df):
    out = survival.univariate_cox(['x'], surv_df, 't', 'e')
    assert out.loc[0, 'n'] == len(surv_df)
    assert out.loc[0, 'events'] == surv_df['e'].sum()


def test_survplot_median_survival_with_quantile(surv_df):
    out = si.survplot('t', 'e', 'g', data=surv_df, output='median_survival', quantile=0.75)
    assert list(out.columns) == ['variable', 'group', 'median survival']


def test_cache_warns_when_query_changed(tmp_path):
    cache = str(tmp_path / 'x.pkl')
    pd.DataFrame({'a': [1]}).to_pickle(cache)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        si.read_sql('SELECT 1', cache=cache)
    with open(cache + '.sql', 'w') as f:
        f.write('SELECT 1')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        si.read_sql('SELECT 1', cache=cache)
    with pytest.warns(UserWarning, match='different query'):
        si.read_sql('SELECT 2', cache=cache)


def test_execute_sql_requires_server():
    with pytest.raises(TypeError):
        si.execute_sql('DELETE FROM t')
