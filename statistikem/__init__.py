from . import comparisons, helpers, correlations, descriptions, survival, sql
from .comparisons import compare, compare_one
from .helpers import format_p, format_value, get_summary, stars, highlight, table_for_mail
from .sql import read_sql, execute_sql
from .correlations import correlate
from .descriptions import describe
from .survival import survplot, survplots

import importlib
import pkgutil

def reload_package():
    """Reload every submodule of statistikem in-place."""
    submodules = [importlib.import_module(f'.{name}', __name__)
                  for _, name, _ in pkgutil.iter_modules(__path__)]
    # Two passes so consumers re-bind to freshly reloaded leaf deps (helpers, sql).
    for _ in range(2):
        for module in submodules:
            importlib.reload(module)
