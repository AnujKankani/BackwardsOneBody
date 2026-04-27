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
