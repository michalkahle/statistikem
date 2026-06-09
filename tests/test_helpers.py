import numpy as np
import pandas as pd
import pytest

from statistikem import helpers


class TestFormatP:
    def test_below_threshold(self):
        assert helpers.format_p(0.0005) == "<0.001"

    def test_three_decimals_between_thresholds(self):
        assert helpers.format_p(0.0123) == "0.012"

    def test_two_decimals_above_point_two(self):
        assert helpers.format_p(0.42) == "0.42"

    def test_exact_one(self):
        assert helpers.format_p(1.0) == "1.0"

    def test_nan_returns_empty(self):
        assert helpers.format_p(float("nan")) == ""

    def test_rejects_p_above_one(self):
        with pytest.raises(ValueError):
            helpers.format_p(1.5)

    def test_iterable_returns_list(self):
        out = helpers.format_p([0.0005, 0.04, 0.5])
        assert out == ["<0.001", "0.040", "0.50"]

    def test_nejm_style_two_vs_three_decimals(self):
        assert helpers.format_p(0.04, style="NEJM") == "0.04"
        assert helpers.format_p(0.005, style="NEJM") == "0.005"
        assert helpers.format_p(0.0001, style="NEJM") == "<0.001"


class TestFormatValue:
    def test_small_int(self):
        assert helpers.format_value(42) == "42.00"

    def test_big_number_uses_g(self):
        assert "e" in helpers.format_value(1.5e6).lower() or helpers.format_value(1.5e6).startswith("1.")

    def test_datetime_returns_date_string(self):
        ts = pd.Timestamp("2024-05-14")
        assert helpers.format_value(ts) == "2024-05-14"

    def test_timedelta_days(self):
        td = pd.Timedelta(days=3)
        assert helpers.format_value(td).endswith(" d")

    def test_timedelta_seconds(self):
        td = pd.Timedelta(seconds=12)
        assert helpers.format_value(td).endswith(" s")


class TestStars:
    @pytest.mark.parametrize(
        "p,expected",
        [
            (0.0001, "***"),
            (0.005, "**"),
            (0.03, "*"),
            (0.2, ""),
        ],
    )
    def test_thresholds(self, p, expected):
        assert helpers.stars(p) == expected

    def test_iterable(self):
        assert helpers.stars([0.0001, 0.5]) == ["***", ""]


class TestGetSummary:
    @pytest.fixture
    def sample(self):
        return pd.Series(np.arange(1, 11), dtype=float)

    def test_iqr(self, sample):
        result = helpers.get_summary(sample, "median (IQR)")
        assert "(" in result and ")" in result

    def test_range(self, sample):
        result = helpers.get_summary(sample, "median (range)")
        assert "(" in result and ")" in result

    def test_five_numbers_returns_list(self, sample):
        result = helpers.get_summary(sample, "5 numbers")
        assert isinstance(result, list)
        assert len(result) == 5

    def test_unknown_summary_raises(self, sample):
        with pytest.raises(ValueError):
            helpers.get_summary(sample, "something else")


class TestGuessScale:
    def test_binary_two_values(self):
        s = pd.Series([0, 1] * 50)
        assert helpers.guess_scale(s) == "binary"

    def test_binary_with_nan(self):
        s = pd.Series([0, 1, np.nan] * 30)
        assert helpers.guess_scale(s) == "binary"

    def test_categorical_low_cardinality(self):
        s = pd.Series(list("abcde") * 20)
        assert helpers.guess_scale(s) == "categorical"

    def test_continuous_high_cardinality(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(size=200))
        assert helpers.guess_scale(s) == "continuous"

    def test_datetime_high_cardinality(self):
        s = pd.Series(pd.date_range("2024-01-01", periods=200, freq="D"))
        assert helpers.guess_scale(s) == "datetime"


class TestCiMeanBootstrap:
    def test_returns_dataframe_by_default(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(size=200))
        out = helpers.ci_mean_bootstrap(s)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["mean", "ci_l", "ci_h"]
        assert out.loc[0, "ci_l"] < out.loc[0, "mean"] < out.loc[0, "ci_h"]

    def test_as_df_false_returns_tuple(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(size=200))
        out = helpers.ci_mean_bootstrap(s, as_df=False)
        assert isinstance(out, tuple)
        assert len(out) == 3
        mean, lo, hi = out
        assert lo < mean < hi

    def test_higher_level_widens_interval(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(size=200))
        _, lo90, hi90 = helpers.ci_mean_bootstrap(s, level=0.90, as_df=False)
        _, lo99, hi99 = helpers.ci_mean_bootstrap(s, level=0.99, as_df=False)
        assert (hi99 - lo99) > (hi90 - lo90)


class TestCiMeanNormal:
    def test_centered_on_mean(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(loc=5, size=500))
        mean, lo, hi = helpers.ci_mean_normal(s)
        assert lo < mean < hi
        assert abs(mean - 5) < 0.5
