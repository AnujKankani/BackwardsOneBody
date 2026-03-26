import logging

from . import BOB_utils
from . import gen_utils
from . import convert_to_strain_using_series
from . import mismatch_utils
from . import BOB_terms
from . import BOB_terms_jax
from . import ascii_funcs

__all__ = ["BOB_utils", "gen_utils", "convert_to_strain_using_series", "mismatch_utils", "BOB_terms", "BOB_terms_jax", "ascii_funcs"]
__version__ = "0.0.1"

# Standard library practice: gwBOB is silent by default so it never
# pollutes output in scripts or notebooks that don't opt in.
logging.getLogger("gwBOB").addHandler(logging.NullHandler())


def enable_output(verbose=False):
    """
    Enable gwBOB console output. Call this once at the top of your script.

    By default gwBOB produces no output so it does not interfere with
    other packages. Calling this function turns on gwBOB-specific
    messages only — other packages (sxs, kuibit, scipy, ...) are
    unaffected.

    Parameters
    ----------
    verbose : bool
        If False (default), show key results such as the waveform
        mismatch and data-loading progress.
        If True, also show internal diagnostic messages useful for
        debugging fits and optimizer behaviour.

    Examples
    --------
    >>> import gwBOB
    >>> gwBOB.enable_output()           # show progress and results
    >>> gwBOB.enable_output(verbose=True)  # also show debug detail
    """
    gwbob_logger = logging.getLogger("gwBOB")

    # Only add a handler if one hasn't been added already (guards against
    # calling enable_output() more than once).
    if not any(isinstance(h, logging.StreamHandler) and
               not isinstance(h, logging.FileHandler)
               for h in gwbob_logger.handlers):
        handler = logging.StreamHandler()
        # Clean format: just the message, no timestamps or module names,
        # so output reads naturally for non-expert users.
        handler.setFormatter(logging.Formatter("%(message)s"))
        gwbob_logger.addHandler(handler)

    gwbob_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Prevent messages from bubbling up to the root logger, which would
    # cause them to appear via any root-level handler other packages may
    # have installed.
    gwbob_logger.propagate = False