import numpy as np
import pandas as pd
import pytest

from statistikem import descriptions


@pytest.fixture
def mixed_frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.normal(40, 10, 30),
            "sex": rng.integers(0, 2, 30),
            "grade": rng.choice(list("abc"), 30),
        }
    )


class TestDescribe:
    def test_returns_one_row_per_column(self, mixed_frame):
        out = descriptions.describe(mixed_frame, plot=False)
        assert len(out) == mixed_frame.shape[1]
        assert set(out.columns) >= {"var", "scale", "description"}

    def test_series_input_treated_as_one_var(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(size=100), name="x")
        out = descriptions.describe(s, plot=False)
        assert len(out) == 1

    def test_parametric_list_broadcasts_to_column_count(self, mixed_frame):
        out = descriptions.describe(
            mixed_frame, parametric=[True], plot=False
        )
        assert len(out) == mixed_frame.shape[1]

    def test_parametric_per_column(self, mixed_frame):
        out = descriptions.describe(
            mixed_frame,
            parametric=[True, None, None],
            plot=False,
        )
        age_row = out[out["var"] == "age"].iloc[0]
        assert "±" in age_row["description"]
