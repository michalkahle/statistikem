from . import comparisons, helpers, correlations, descriptions, survival
from .comparisons import compare, compare_one
from .helpers import format_p, format_float, stars
from .correlations import correlate
from .descriptions import describe
from .survival import kmplot, kmplots, kmtable

import importlib

def reload_package():
    """
    Reloads all modules in the package.
    """
    importlib.reload(helpers)
    importlib.reload(comparisons)
    importlib.reload(helpers)
    importlib.reload(correlations)
    importlib.reload(descriptions)
    importlib.reload(survival)