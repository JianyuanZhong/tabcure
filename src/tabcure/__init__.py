try:
    from ._version import __version__ as __version__
except ImportError:
    import sys

    print(
        "Project not installed in the current env, activate the correct env or install it with:\n\tpip install -e .",
        file=sys.stderr,
    )
    __version__ = "unknown"

from . import tabular_metrices as Metrices
from .modules import TabCure, seed_everything, set_logging_level

__all__ = ["TabCure", "Metrices", "seed_everything", "set_logging_level"]
