"""
Download the SXS and CCE waveform data required by gwBOB's integration tests.

The unit tests in ``tests/unit/`` need no external data and run without this
script. The integration tests, regression baselines, and trusted-output
comparisons in ``tests/integration/`` need:

  - SXS:BBH:2325 (extrapolated waveforms from the SXS catalog) — fetched
    via the ``sxs`` package with ``SXSCACHEDIR`` pointed at ``tests/sxs_cache/``.
  - SXS:BBH_ExtCCE:0009 (the "cce9" simulation) — fetched directly from the
    Zenodo record (10.5281/zenodo.10783596) via ``urlretrieve``. Only the
    six ``Lev5:*_BondiCce_R0270.h5`` files plus their JSON sidecars are
    pulled; that's everything ``BOB.initialize_with_cce_data`` consumes.

Run it once after cloning::

    cd BackwardsOneBody/
    python tests/fetch_data.py

The script is idempotent — re-running with the cache already populated is
a quick no-op. Total download is ~90 MB.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


TESTS_DIR = Path(__file__).resolve().parent
CACHE_DIR = TESTS_DIR / "sxs_cache"

# Zenodo intermittently returns 502/503/504 from its gateway, and a single
# blip used to abort the whole ~90 MB fetch (CI hit a 504 on the very first
# CCE file). Retry transient failures with exponential backoff; permanent
# ones (404, 403) still fail immediately so a genuinely wrong URL is loud.
DOWNLOAD_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 2.0
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
# Without this a stalled connection hangs forever; urlretrieve has no timeout
# argument, so it has to be set globally.
SOCKET_TIMEOUT_SECONDS = 60.0

SXS_BBH_ID = "SXS:BBH:2325"
CCE_ZENODO_RECORD = "10783596"
CCE_LEVEL = 5
CCE_RADIUS = 270
CCE_SIM_NAME = "SXS:BBH_ExtCCE:0009"
CCE_LOCAL_DIRNAME = "cce9"

# Match the ordering / filenames used by ``qnmfits.cce`` so existing test
# fixtures find the data at the expected paths.
CCE_WAVEFORM_TYPES = (
    "rhOverM",
    "rMPsi4",
    "r2Psi3",
    "r3Psi2OverM",
    "r4Psi1OverM2",
    "r5Psi0OverM3",
)


def _download_if_missing(url: str, dest: Path) -> bool:
    """Download ``url`` to ``dest`` unless the file already exists.

    Returns True if a download happened, False if the file was already cached.
    """
    if dest.is_file() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    socket.setdefaulttimeout(SOCKET_TIMEOUT_SECONDS)

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            urlretrieve(url, tmp)
            tmp.rename(dest)
            return True
        except BaseException as exc:
            # Don't leave a half-written .part file behind on Ctrl-C, a network
            # error, or a failed attempt we're about to retry.
            if tmp.exists():
                tmp.unlink()

            transient = (
                isinstance(exc, HTTPError) and exc.code in RETRYABLE_STATUS
            ) or (
                # URLError covers DNS failures and connection resets; a bare
                # socket.timeout can surface mid-transfer. Neither is fatal.
                isinstance(exc, (URLError, socket.timeout))
                and not isinstance(exc, HTTPError)
            )
            if not transient or attempt == DOWNLOAD_ATTEMPTS:
                raise

            delay = RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
            print(
                f"    attempt {attempt}/{DOWNLOAD_ATTEMPTS} failed ({exc}); "
                f"retrying in {delay:.0f}s ...",
                flush=True,
            )
            time.sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise AssertionError("retry loop exited without returning")


def fetch_sxs_bbh_2325() -> None:
    """Populate ``tests/sxs_cache/`` with SXS:BBH:2325 via the ``sxs`` package.

    ``sxs.load`` honors ``SXSCACHEDIR`` so the data lands inside the test
    cache rather than the user's home directory. With ``download=True`` it
    is a no-op when the simulation is already present.
    """
    print(f"[1/2] SXS catalog: {SXS_BBH_ID}")
    os.environ["SXSCACHEDIR"] = str(CACHE_DIR)
    os.environ["SXSCONFIGDIR"] = str(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import sxs

    # The cache may contain a config.json with ``{"download": false}`` (set by
    # tests that want offline behavior). That setting overrides the
    # ``download=True`` kwarg on lazy property accesses below, so flip it
    # while we fetch and restore on the way out.
    try:
        prior_download = sxs.read_config("download", True)
    except Exception:
        prior_download = True
    sxs.write_config(download=True)
    try:
        sim = sxs.load(SXS_BBH_ID, download=True)
        # ``sxs.load`` is lazy — it only downloads each file when its
        # property is first accessed. Force-fetch the two waveform files
        # the integration tests use: ``sim.h`` pulls Lev3:Strain_N2.{h5,json}
        # and ``sim.psi4`` pulls Lev3:ExtraWaveforms.{h5,json}. Skipping
        # psi4 caused CI to fail with "ExtraWaveforms.json not found and
        # download is disabled" once the test reached ``sim.psi4``.
        _ = sim.h
        _ = sim.psi4
    finally:
        sxs.write_config(download=prior_download)
    print(f"  cached at {CACHE_DIR}/SXS:BBH:2325v*/")


def fetch_cce9() -> None:
    """Download the cce9 (``SXS:BBH_ExtCCE:0009``) waveform files from Zenodo.

    Replicates the URL pattern used by ``qnmfits.cce.load`` so the resulting
    layout matches what the ``BOB_cce`` fixture expects, but writes directly
    to ``tests/sxs_cache/cce9/`` instead of the qnmfits package directory.
    """
    print(f"[2/2] Zenodo record {CCE_ZENODO_RECORD}: {CCE_SIM_NAME} (Lev{CCE_LEVEL})")
    base_url = f"https://zenodo.org/records/{CCE_ZENODO_RECORD}/files"
    dest_dir = CACHE_DIR / CCE_LOCAL_DIRNAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    metadata_dest = dest_dir / "metadata.json"
    # Note: Zenodo serves metadata.json under ``Lev{level}/metadata.json``
    # (with a slash) while H5 + sidecar JSON files use ``Lev{level}:<name>``
    # (with a colon). This mirrors the URL pattern in qnmfits.cce.load.
    metadata_url = f"{base_url}/Lev{CCE_LEVEL}/metadata.json?download=1"
    downloaded_metadata = _download_if_missing(metadata_url, metadata_dest)

    n_downloaded = int(downloaded_metadata)
    for wf in CCE_WAVEFORM_TYPES:
        h5_name = f"{wf}_BondiCce_R{CCE_RADIUS:04d}.h5"
        json_name = f"{wf}_BondiCce_R{CCE_RADIUS:04d}.json"
        h5_url = f"{base_url}/Lev{CCE_LEVEL}:{h5_name}?download=1"
        json_url = f"{base_url}/Lev{CCE_LEVEL}:{json_name}?download=1"
        if _download_if_missing(h5_url, dest_dir / h5_name):
            n_downloaded += 1
        if _download_if_missing(json_url, dest_dir / json_name):
            n_downloaded += 1

    if n_downloaded == 0:
        print(f"  already cached at {dest_dir}/")
    else:
        print(f"  downloaded {n_downloaded} file(s) to {dest_dir}/")


def main() -> int:
    print("gwBOB test data fetcher")
    print(f"cache directory: {CACHE_DIR}")
    print()
    try:
        fetch_sxs_bbh_2325()
        fetch_cce9()
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run to resume; partial files were cleaned up.", file=sys.stderr)
        return 130
    print()
    print("Done. Run integration tests with: pytest tests/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
