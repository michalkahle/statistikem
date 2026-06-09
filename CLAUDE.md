# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`statistikem` is a Python library for quick statistical assessment of variables in small datasets. It provides descriptive statistics, group comparisons, correlation analysis, and survival analysis — all with automatic scale detection and integrated plotting.

Installed in editable mode (`pip install -e .`). After editing source files, call `statistikem.reload_package()` in a running Python/Jupyter session to pick up changes without restarting the kernel.

## Development Commands

```bash
# Install in editable mode
pip install -e .

# No test suite exists — functionality is validated interactively via statistikem.ipynb
```

## Architecture

The package lives entirely in `statistikem/`. Each module is independently importable; `__init__.py` re-exports the public API.

### Module responsibilities

| Module | Key exports | Purpose |
|---|---|---|
| `comparisons.py` | `compare()`, `compare_one()` | Core group-comparison engine |
| `descriptions.py` | `describe()` | Descriptive statistics for a DataFrame or Series |
| `correlations.py` | `correlate()`, `plot_correlation()` | Correlation matrices and scatter plots |
| `survival.py` | `survplot()`, `survplots()`, `cox()`, `univariate_cox()` | Kaplan-Meier, Aalen-Johansen, Cox regression |
| `helpers.py` | `guess_scale()`, `format_p()`, `read_sql()`, etc. | Shared utilities used by all other modules |

### Central design pattern: automatic scale detection

`helpers.guess_scale(series)` classifies every variable as `"binary"`, `"categorical"`, or `"continuous"`. All comparison and description functions call this to pick the right statistical test and plot type automatically, so callers rarely need to specify a scale manually.

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

`helpers.read_sql()` and `helpers.execute_sql()` connect via `jaydebeapi`. Credentials/connection strings are passed at call time — nothing is stored in the package.
