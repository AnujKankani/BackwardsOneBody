"""
Shared pytest configuration and fixtures for the gwBOB test suite.

Path resolution and SXS cache configuration are centralized here so individual
test modules don't have to reach for ``sys.path`` hacks or hard-coded ``./tests``
prefixes. Adding a new test module just needs to import a fixture by name.

See ``DESIGN_test_refactor.md`` for the design rationale.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Path fixtures
#
# All paths are resolved relative to the directory containing this file
# (``tests/``). Tests work regardless of the current working directory.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tests_dir() -> Path:
    """Absolute path to the ``tests/`` directory."""
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def trusted_outputs_dir(tests_dir: Path) -> Path:
    """Absolute path to ``tests/trusted_outputs/`` (reference NPZ files)."""
    return tests_dir / "trusted_outputs"


@pytest.fixture(scope="session")
def sxs_cache_dir(tests_dir: Path) -> Path:
    """Absolute path to ``tests/sxs_cache/`` (cached SXS / CCE simulation data)."""
    return tests_dir / "sxs_cache"


# ---------------------------------------------------------------------------
# SXS cache configuration
#
# The ``sxs`` package reads ``SXSCACHEDIR`` / ``SXSCONFIGDIR`` env vars at
# import time to decide where to look for cached simulations. Set them once,
# session-wide, before any test imports ``sxs``.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _configure_sxs_cache(sxs_cache_dir: Path) -> None:
    """Auto-applied: point the ``sxs`` package at our committed test cache."""
    # ``setdefault`` preserves any value the user explicitly set in their shell
    # (e.g., when running locally outside of pytest's CI invocation).
    os.environ.setdefault("SXSCACHEDIR", str(sxs_cache_dir))
    os.environ.setdefault("SXSCONFIGDIR", str(sxs_cache_dir))


# ---------------------------------------------------------------------------
# Conditional skips for missing test data
#
# Some test data (notably the cce9 CCE simulation files) is large and may not
# always be present locally. Tests that require it can request these fixtures
# and skip cleanly when the data is missing.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cce9_available(sxs_cache_dir: Path) -> bool:
    """True iff the cce9 simulation files are present."""
    required = [
        "cce9/rhOverM_BondiCce_R0270.h5",
        "cce9/rMPsi4_BondiCce_R0270.h5",
        "cce9/r2Psi3_BondiCce_R0270.h5",
        "cce9/r3Psi2OverM_BondiCce_R0270.h5",
        "cce9/r4Psi1OverM2_BondiCce_R0270.h5",
        "cce9/r5Psi0OverM3_BondiCce_R0270.h5",
    ]
    return all((sxs_cache_dir / p).exists() for p in required)


@pytest.fixture(scope="session")
def sxs_bbh_2325_available(sxs_cache_dir: Path) -> bool:
    """True iff the SXS:BBH:2325 simulation files are present."""
    return (sxs_cache_dir / "SXS:BBH:2325v3.0").is_dir()


# ---------------------------------------------------------------------------
# Helper: load a numpy reference file with a context manager (closes the
# file descriptor as soon as the dict is built).
# ---------------------------------------------------------------------------

def load_npz_dict(path: Path) -> dict:
    """Load all arrays from an NPZ file into a plain dict.

    Uses a context manager so the underlying file descriptor is closed
    promptly (addresses code review §4 P3 E9).
    """
    import numpy as np
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


# ---------------------------------------------------------------------------
# Synthetic fixtures for unit tests (T4 of test refactor)
#
# These let unit tests exercise gwBOB internals without loading any NR data.
# They're deliberately minimal: just enough surface area to satisfy the
# function-under-test, with deterministic values so reference outputs can be
# computed by hand or from documented analytic formulas.
# ---------------------------------------------------------------------------

class _MockMultiModeStrain:
    """Minimal stand-in for ``sxs.WaveformModes`` (strain-style attribute access).

    Supports the API that ``gen_utils.get_kuibit_lm`` uses:
      - ``w.t``                : 1D time array
      - ``w.data``             : 2D complex array (n_samples, n_modes)
      - ``w.index(l, m)``      : column index for the (l, m) mode
    """
    def __init__(self, t, data, lm_to_index):
        import numpy as np
        self.t = np.asarray(t)
        self.data = np.asarray(data)
        self._lm_to_index = dict(lm_to_index)

    def index(self, l, m):
        return self._lm_to_index[(int(l), int(m))]


class _MockMultiModePsi4Slice:
    """Slice helper used by ``_MockMultiModePsi4`` — exposes ``.ndarray``."""
    def __init__(self, arr):
        self.ndarray = arr


class _MockMultiModePsi4:
    """Minimal stand-in for the psi4 attribute-access pattern.

    Supports the API that ``gen_utils.get_kuibit_lm_psi4`` uses:
      - ``w.t``                : 1D time array
      - ``w.index(l, m)``      : column index for the (l, m) mode
      - ``w[:, idx].ndarray``  : 1D array of mode values
    """
    def __init__(self, t, data, lm_to_index):
        import numpy as np
        self.t = np.asarray(t)
        self._data = np.asarray(data)
        self._lm_to_index = dict(lm_to_index)

    def index(self, l, m):
        return self._lm_to_index[(int(l), int(m))]

    def __getitem__(self, key):
        return _MockMultiModePsi4Slice(self._data[key])


@pytest.fixture
def synthetic_t():
    """A regular time grid: ``np.arange(-50, 50, 0.1)``."""
    import numpy as np
    return np.arange(-50.0, 50.0, 0.1)


@pytest.fixture
def synthetic_kuibit_ts(synthetic_t):
    """A complex timeseries with a known constant angular frequency.

    Encodes ``y(t) = exp(-i * omega * t)`` with ``omega = 0.2``. Useful for
    asserting that frequency / phase extraction recover the encoded value.
    """
    from kuibit.timeseries import TimeSeries as kuibit_ts
    import numpy as np
    omega = 0.2
    y = np.exp(-1j * omega * synthetic_t)
    ts = kuibit_ts(synthetic_t, y)
    ts._encoded_omega = omega   # so tests can refer back to the truth
    return ts


@pytest.fixture
def synthetic_multimode_strain():
    """A small multi-mode strain-like object with deterministic per-mode data.

    3 samples in time, 4 (l, m) modes — kept tiny so reference outputs are
    obvious. Mode (l, m) at sample i has value ``i + 100*l + m`` (real),
    ``-i - 100*l - m`` (imag) — so we can identify which column a slice came
    from by inspection.
    """
    import numpy as np
    t = np.array([0.0, 1.0, 2.0])
    lm_to_index = {(2, 2): 0, (2, -2): 1, (3, 3): 2, (3, -3): 3}
    n_samples, n_modes = len(t), len(lm_to_index)
    data = np.zeros((n_samples, n_modes), dtype=np.complex128)
    for (l, m), idx in lm_to_index.items():
        for i in range(n_samples):
            data[i, idx] = (i + 100 * l + m) - 1j * (i + 100 * l + m)
    return _MockMultiModeStrain(t, data, lm_to_index)


@pytest.fixture
def synthetic_multimode_psi4(synthetic_multimode_strain):
    """A psi4-style multi-mode object with the same shape/data as the strain
    fixture. Differs only in attribute-access pattern (``w[:, idx].ndarray``)."""
    return _MockMultiModePsi4(
        synthetic_multimode_strain.t,
        synthetic_multimode_strain.data,
        synthetic_multimode_strain._lm_to_index,
    )


@pytest.fixture
def synthetic_bob():
    """A ``SimpleNamespace`` mimicking the attribute surface of a ``BOB``
    instance — just enough for ``BOB_terms.*`` to read.

    Designed for: ``Omega_0 < Omega_QNM`` (always true for physical remnants),
    ``tau > 0``, peak time ``tp = 0`` so ``t_tp_tau = t / tau``.

    The defaults below are arbitrary-but-physical: representative for a
    near-equal-mass BBH remnant.
    """
    from types import SimpleNamespace
    import numpy as np

    Omega_0   = 0.15
    Omega_QNM = 0.4
    Phi_0     = 0.0
    tau       = 10.0
    tp        = 0.0
    t0        = -10.0
    Ap        = 0.1
    m         = 2

    # Wide, symmetric window so asymptotic tests have plenty of "tail" on
    # both sides: |t / tau| reaches 10 at the edges, and tanh(±10) ≈ ±1 to
    # ~9 decimal places, so the asymptotic frequency limits are essentially
    # exact. ``linspace`` (vs ``arange``) guarantees an odd-length symmetric
    # grid with t = 0 exactly at the centre.
    t = np.linspace(-100.0, 100.0, 2001)
    t_tp_tau  = (t - tp) / tau
    t0_tp_tau = (t0 - tp) / tau

    return SimpleNamespace(
        Omega_0=Omega_0, Omega_QNM=Omega_QNM, Phi_0=Phi_0,
        tau=tau, tp=tp, t0=t0,
        Ap=Ap, m=m,
        t=t, t_tp_tau=t_tp_tau, t0_tp_tau=t0_tp_tau,
        # Some BOB_terms.* finite_t0 phase functions try the analytic form
        # and, on ValueError, consult ``auto_switch_to_numerical_integration``
        # before deciding whether to fall back to a numerical antiderivative.
        # Real ``BOB`` instances default this to True.
        auto_switch_to_numerical_integration=True,
    )


@pytest.fixture
def synthetic_bob_finite(synthetic_bob):
    """A bob whose time grid starts AT ``t0`` and extends far into ``t >> tp``.

    This is the valid range for finite-t0 BOB_terms formulas: at ``t < t0``,
    psi4_finite_t0 produces a negative radicand (raises) and
    news/strain_finite_t0 produce unphysical values. Use this fixture for any
    test of the ``BOB_*_finite_t0`` family.
    """
    from types import SimpleNamespace
    import numpy as np

    t0  = synthetic_bob.t0
    tp  = synthetic_bob.tp
    tau = synthetic_bob.tau
    # Range [t0, t0 + 110] — covers t = t0 plus a long tail well past tp.
    t = np.linspace(t0, t0 + 110.0, 1101)
    return SimpleNamespace(
        Omega_0=synthetic_bob.Omega_0,
        Omega_QNM=synthetic_bob.Omega_QNM,
        Phi_0=synthetic_bob.Phi_0,
        tau=tau, tp=tp, t0=t0,
        Ap=synthetic_bob.Ap, m=synthetic_bob.m,
        t=t,
        t_tp_tau=(t  - tp) / tau,
        t0_tp_tau=(t0 - tp) / tau,
        auto_switch_to_numerical_integration=True,
    )


# ---------------------------------------------------------------------------
# Domain fixtures
#
# Heavyweight fixtures that load real NR data. Tests that need them request
# them by name; the fixtures skip cleanly when the underlying data files are
# not on disk.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def BOB_cce(sxs_cache_dir: Path, cce9_available: bool):
    """A session-scoped BOB initialized from cached cce9 CCE data.

    Skips cleanly if any of the cce9 simulation files are missing.
    """
    if not cce9_available:
        pytest.skip("cce9 simulation files not present in tests/sxs_cache/cce9/")

    # Imports are local so test collection works even when scri / kuibit are
    # missing or import-slow on a stripped CI image.
    import scri
    from gwBOB import BOB_utils

    cce9 = sxs_cache_dir / "cce9"
    wf_paths = {
        "h":    str(cce9 / "rhOverM_BondiCce_R0270.h5"),
        "Psi4": str(cce9 / "rMPsi4_BondiCce_R0270.h5"),
        "Psi3": str(cce9 / "r2Psi3_BondiCce_R0270.h5"),
        "Psi2": str(cce9 / "r3Psi2OverM_BondiCce_R0270.h5"),
        "Psi1": str(cce9 / "r4Psi1OverM2_BondiCce_R0270.h5"),
        "Psi0": str(cce9 / "r5Psi0OverM3_BondiCce_R0270.h5"),
    }

    abd = scri.SpEC.create_abd_from_h5(file_format="RPDMB", **wf_paths)
    bob = BOB_utils.BOB()
    bob.initialize_with_cce_data(-1, provide_own_abd=abd, l=2, m=-2)
    return bob


@pytest.fixture(scope="session")
def initial_sxs_bob_2325(sxs_bbh_2325_available):
    """A BOB initialized with SXS:BBH:2325 — loaded ONCE per test session.

    Tests that need fresh state should ``copy.deepcopy()`` this fixture
    rather than calling ``initialize_with_sxs_data`` themselves; loading
    the SXS waveform is expensive (~50 MB allocations) and doing it many
    times in one process exhausts memory in constrained environments
    such as WSL. See CLAUDE.md "Memory awareness".
    """
    if not sxs_bbh_2325_available:
        pytest.skip("SXS:BBH:2325 cache not present in tests/sxs_cache/")

    from gwBOB import BOB_utils
    bob = BOB_utils.BOB()
    bob.initialize_with_sxs_data("SXS:BBH:2325", l=2, m=2, download=False)
    return bob
