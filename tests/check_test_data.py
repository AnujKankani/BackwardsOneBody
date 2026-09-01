#!/usr/bin/env python
"""Report whether the NR test data needed by the integration tests is present.

Exit code 0 if every required path exists and is non-empty, 1 otherwise (with
the missing paths listed on stderr).

Why this exists: the ``cce9_available`` / ``sxs_bbh_2325_available`` fixtures in
``tests/conftest.py`` make the integration tests ``pytest.skip()`` when the data
is absent, and pytest reports skips as green. So a CI run with an incomplete
data cache could pass without executing a single integration test. CI calls this
script twice — once to decide whether a fetch is needed, and once after the
fetch as a hard gate — so that outcome is impossible.

    python tests/check_test_data.py            # from BackwardsOneBody/

It lives in ``tests/`` next to ``fetch_data.py`` (which it pairs with) rather
than in ``scripts/`` because ``scripts/`` is gitignored — CI needs this file to
exist in a clone. It is not a test and pytest does not collect it (no ``test_``
prefix).

Claude Code: the required-path list below must stay in sync with the fixtures in
tests/conftest.py. It is deliberately an exact mirror rather than a looser glob:
``sxs_bbh_2325_available`` checks for the literal ``SXS:BBH:2325v3.0`` directory,
so accepting e.g. ``SXS:BBH:2325v3.1`` here would let the check pass while the
fixture still returned False — reintroducing the silent-skip hole this guards.
"""

from __future__ import annotations

import sys
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "sxs_cache"

# Mirrors tests/conftest.py::cce9_available
CCE9_FILES = (
    "cce9/rhOverM_BondiCce_R0270.h5",
    "cce9/rMPsi4_BondiCce_R0270.h5",
    "cce9/r2Psi3_BondiCce_R0270.h5",
    "cce9/r3Psi2OverM_BondiCce_R0270.h5",
    "cce9/r4Psi1OverM2_BondiCce_R0270.h5",
    "cce9/r5Psi0OverM3_BondiCce_R0270.h5",
)

# Mirrors tests/conftest.py::sxs_bbh_2325_available
SXS_DIR = "SXS:BBH:2325v3.0"


def missing_paths() -> list[str]:
    """Return the required paths that are absent or empty."""
    missing = [
        p for p in CCE9_FILES
        if not (CACHE / p).is_file() or (CACHE / p).stat().st_size == 0
    ]
    if not (CACHE / SXS_DIR).is_dir():
        missing.append(SXS_DIR + "/")
    return missing


def main() -> int:
    missing = missing_paths()
    if missing:
        print("NR test data is incomplete.", file=sys.stderr)
        for m in missing:
            print(f"  missing: {m}", file=sys.stderr)
        print(
            "Integration tests would SKIP (and report green) without it.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: all {len(CCE9_FILES) + 1} required test-data paths present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
