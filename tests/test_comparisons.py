import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from statistikem import comparisons


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def normal_two_group(rng):
    n = 50
    return pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, n), rng.normal(1, 1, n)]),
            "g": ["a"] * n + ["b"] * n,
        }
    )


@pytest.fixture
def skewed_two_group(rng):
    n = 80
    return pd.DataFrame(
        {
            "x": np.concatenate(
                [rng.exponential(1.0, n), rng.exponential(2.0, n)]
            ),
            "g": ["a"] * n + ["b"] * n,
        }
    )


@pytest.fixture
def binary_two_group(rng):
    n = 80
    return pd.DataFrame(
        {
            "y": np.concatenate(
                [rng.binomial(1, 0.3, n), rng.binomial(1, 0.6, n)]
            ),
            "g": ["a"] * n + ["b"] * n,
        }
    )


@pytest.fixture
def paired_continuous(rng):
    n = 40
    subject = np.arange(n)
    before = rng.normal(0, 1, n)
    after = before + rng.normal(0.5, 0.5, n)
    return pd.DataFrame(
        {
            "value": np.concatenate([before, after]),
            "time": ["before"] * n + ["after"] * n,
            "subject": np.concatenate([subject, subject]),
        }
    )


@pytest.fixture
def paired_binary(rng):
    n = 60
    subject = np.arange(n)
    before = rng.binomial(1, 0.3, n)
    flip = rng.binomial(1, 0.4, n)
    after = np.where(flip, 1 - before, before)
    return pd.DataFrame(
        {
            "y": np.concatenate([before, after]),
            "time": ["before"] * n + ["after"] * n,
            "subject": np.concatenate([subject, subject]),
        }
    )


class TestCompareOneIndependentContinuous:
    def test_normal_picks_t(self, normal_two_group):
        res = comparisons.compare_one(
            "x", "g", data=normal_two_group, plot=False
        )
        assert res["test"] == "t"
        assert res["scale"] == "continuous"
        assert "a" in res and "b" in res

    def test_force_nonparametric(self, normal_two_group):
        res = comparisons.compare_one(
            "x", "g", data=normal_two_group, plot=False, parametric=False
        )
        assert res["test"] == "Mann-Whitney"

    def test_skewed_picks_nonparametric(self, skewed_two_group):
        res = comparisons.compare_one(
            "x", "g", data=skewed_two_group, plot=False
        )
        assert res["test"] == "Mann-Whitney"


class TestCompareOneIndependentBinary:
    def test_two_by_two_runs_fisher(self, binary_two_group):
        res = comparisons.compare_one(
            "y", "g", data=binary_two_group, plot=False
        )
        assert res["test"] in ("Fisher exact", "Yates chi^2")
        assert res["scale"] == "binary"
        assert 0 <= res["p"] <= 1


class TestCompareOneIndependentCategorical:
    @pytest.fixture
    def string_categorical(self, rng):
        n = 90
        return pd.DataFrame(
            {
                "grade": rng.choice(["mild", "moderate", "severe"], n),
                "g": ["a"] * (n // 2) + ["b"] * (n // 2),
            }
        )

    def test_string_categorical_uses_contingency_test(self, string_categorical):
        res = comparisons.compare_one(
            "grade", "g", data=string_categorical, plot=False
        )
        assert res["scale"] == "categorical"
        assert res["test"] in ("Pearson chi^2", "Fisher exact")
        assert 0 <= res["p"] <= 1

    def test_per_category_cell_summaries(self, string_categorical):
        res = comparisons.compare_one(
            "grade", "g", data=string_categorical, plot=False
        )
        # every category should appear in each group's cell summary
        for group in ("a", "b"):
            for cat in ("mild", "moderate", "severe"):
                assert cat in res[group]

    def test_batch_categorical_has_real_p(self, string_categorical):
        out = comparisons.compare(
            ["grade"], "g", data=string_categorical, plot=False
        )
        assert out["p"].notna().all()
        assert "p_corr" in out.columns


class TestCompareOnePairedContinuous:
    def test_normal_paired_picks_paired_t(self, paired_continuous):
        res = comparisons.compare_one(
            "value", "time", subject="subject",
            data=paired_continuous, plot=False,
        )
        assert res["test"] == "paired t"
        assert 0 <= res["p"] <= 1

    def test_force_nonparametric(self, paired_continuous):
        res = comparisons.compare_one(
            "value", "time", subject="subject",
            data=paired_continuous, plot=False, parametric=False,
        )
        assert res["test"] == "signed-rank"


class TestCompareOnePairedBinary:
    def test_two_by_two_runs_mcnemar(self, paired_binary):
        res = comparisons.compare_one(
            "y", "time", subject="subject",
            data=paired_binary, plot=False,
        )
        assert res["scale"] == "binary"
        assert res["outcome"] == "time"
        assert res["predictor"] == "y"


class TestCompareBatch:
    def test_restores_figure_max_open_warning(self, normal_two_group):
        original = plt.rcParams["figure.max_open_warning"]
        plt.rcParams["figure.max_open_warning"] = 17
        try:
            comparisons.compare(
                ["x"], "g", data=normal_two_group, plot=False
            )
            assert plt.rcParams["figure.max_open_warning"] == 17
        finally:
            plt.rcParams["figure.max_open_warning"] = original

    def test_skips_grouping_and_subject_columns(self, paired_continuous):
        out = comparisons.compare(
            list(paired_continuous.columns),
            grouping="time",
            subject="subject",
            data=paired_continuous,
            plot=False,
        )
        assert "time" not in out["predictor"].values
        assert "subject" not in out["predictor"].values

    def test_adds_p_corr_column(self, normal_two_group):
        out = comparisons.compare(
            ["x"], "g", data=normal_two_group, plot=False
        )
        assert "p_corr" in out.columns
        assert (out["p_corr"] >= out["p"]).all()


class TestCompareOnePlottingSmoke:
    """plot=True is the interactive default; smoke-test that each branch renders without error."""

    def test_independent_continuous_plots(self, normal_two_group):
        res = comparisons.compare_one(
            "x", "g", data=normal_two_group, plot=True
        )
        assert res["test"] == "t"
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_independent_binary_plots(self, binary_two_group):
        res = comparisons.compare_one(
            "y", "g", data=binary_two_group, plot=True
        )
        assert res["scale"] == "binary"
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_paired_continuous_plots(self, paired_continuous):
        res = comparisons.compare_one(
            "value", "time", subject="subject",
            data=paired_continuous, plot=True,
        )
        assert res["test"] in ("paired t", "signed-rank")
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_paired_binary_plots(self, paired_binary):
        res = comparisons.compare_one(
            "y", "time", subject="subject",
            data=paired_binary, plot=True,
        )
        assert res["scale"] == "binary"
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_independent_categorical_plots(self, rng):
        n = 60
        df = pd.DataFrame({
            "grade": rng.integers(1, 4, n),
            "g": ["a"] * (n // 2) + ["b"] * (n // 2),
        })
        comparisons.compare_one("grade", "g", data=df, plot=True, scale="categorical")
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_independent_string_categorical_plots(self, rng):
        n = 90
        df = pd.DataFrame({
            "grade": rng.choice(["mild", "moderate", "severe"], n),
            "g": ["a"] * (n // 2) + ["b"] * (n // 2),
        })
        comparisons.compare_one("grade", "g", data=df, plot=True)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_paired_categorical_plots(self, rng):
        n = 50
        subject = np.arange(n)
        before = rng.choice(["mild", "moderate", "severe"], n)
        after = rng.choice(["mild", "moderate", "severe"], n)
        df = pd.DataFrame({
            "grade": np.concatenate([before, after]),
            "time": ["before"] * n + ["after"] * n,
            "subject": np.concatenate([subject, subject]),
        })
        res = comparisons.compare_one(
            "grade", "time", subject="subject", data=df, plot=True,
        )
        assert res["scale"] == "categorical"
        import matplotlib.pyplot as plt
        plt.close("all")


class TestPivotPairedDoesNotMutateInput:
    def test_input_index_preserved(self, paired_continuous):
        value = paired_continuous["value"]
        time = paired_continuous["time"]
        subject = paired_continuous["subject"]
        original_index = value.index.copy()
        comparisons._pivot_paired(value, time, subject)
        pd.testing.assert_index_equal(value.index, original_index)
