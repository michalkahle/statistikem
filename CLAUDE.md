# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`statistikem` is a Python library for quick statistical assessment of variables in small datasets. It provides descriptive statistics, group comparisons, correlation analysis, and survival analysis — all with automatic scale detection and integrated plotting.

Installed in editable mode (`pip install -e .`). After editing source files, call `statistikem.reload_package()` in a running Python/Jupyter session to pick up changes without restarting the kernel.

## Development Commands

```bash
# Install in editable mode
pip install -e .

# Run the test suite (pytest)
pytest

# Run a single test file / test
pytest tests/test_comparisons.py
pytest tests/test_helpers.py::test_guess_scale
```

`tests/conftest.py` forces the non-interactive `Agg` matplotlib backend, so tests
that build plots run headless. The tests live in `tests/`; broader exploratory
validation happens interactively in `statistikem.ipynb`.

## Architecture

The package lives entirely in `statistikem/`. Each module is independently importable; `__init__.py` re-exports the public API.

### Module responsibilities

| Module | Key exports | Purpose |
|---|---|---|
| `comparisons.py` | `compare()`, `compare_one()` | Core group-comparison engine |
| `descriptions.py` | `describe()` | Descriptive statistics for a DataFrame or Series |
| `correlations.py` | `correlate()`, `plot_correlation()` | Correlation matrices and scatter plots |
| `survival.py` | `survplot()`, `survplots()`, `cox()`, `univariate_cox()` | Kaplan-Meier, Aalen-Johansen, Cox regression |
| `helpers.py` | `guess_scale()`, `format_p()`, `stars()`, etc. | Shared utilities used by all other modules |
| `sql.py` | `read_sql()`, `execute_sql()` | IRIS DB-API database access |

### Central design pattern: automatic scale detection

`helpers.guess_scale(series)` classifies every variable as `"binary"`, `"categorical"`, or `"numeric"`. All comparison and description functions call this to pick the right statistical test and plot type automatically, so callers rarely need to specify a scale manually.

### Statistical test selection in `comparisons.py`

`compare_one()` dispatches to one of four private helpers depending on scale and paired/independent design:

- `_independent_difference()` — t-test / ANOVA / Mann-Whitney / Kruskal-Wallis (normality tested with Shapiro-Wilk)
- `_paired_difference()` — paired t-test / Wilcoxon signed-rank
- `_independent_proportion()` — chi-squared / Fisher exact
- `_paired_proportion()` — McNemar

`compare()` is the batch wrapper: it calls `compare_one()` for each variable and assembles results into a summary table, applying multiple-testing correction (Holm-Sidak by default).

### P-value formatting

`helpers.format_p()` formats p-values following NEJM conventions (e.g. `<0.001`). The `stars()` helper converts p-values to significance notation (`*`, `**`, `***`).

### Database access

`sql.read_sql()` and `sql.execute_sql()` connect via the native IRIS DB-API driver
(`iris.dbapi`, `intersystems-irispython>=5.4.1`). Credentials resolve in order: explicit
`url`/`user`/`password` arguments, the per-server in-memory cache (`_credentials`), the
`ZLATOKOP_<SERVER>` environment variable (whitespace-separated `url user password`,
`.env` loaded at import), then an interactive prompt. `read_sql(..., cache='path.pkl')`
pickles the result on first call and reads from it thereafter — delete the file to
refresh.
