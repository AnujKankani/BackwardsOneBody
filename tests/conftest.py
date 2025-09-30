# tests/conftest.py (Final, Corrected Version)
import pytest
import sxs
from pathlib import Path

@pytest.fixture(scope="session", autouse=True)
def sxs_test_environment_setup():
    """
    This fixture runs automatically before any tests. It patches the sxs library
    at runtime to ensure it can find the sanitized, pre-downloaded test data.
    It uses a try/finally block to guarantee the original library is restored.
    """
    # --- Setup Phase ---

    # 1. Save the original library function at the very start of the fixture.
    original_function = sxs.utilities.sxs_path_to_system_path

    # 2. Define our new, safe path converter.
    #    It uses the 'original_function' variable from this fixture's scope (a closure).
    def universal_safe_sxs_path_converter(path):
        """A patched version that replaces colons with hyphens, regardless of OS."""
        raw_path = original_function(path)
        return str(raw_path).replace(":", "-").replace("_", "-")

    # 3. Apply the patch.
    sxs.utilities.sxs_path_to_system_path = universal_safe_sxs_path_converter

    # 4. Configure the cache path. This is now the ONLY place this is done.
    local_cache_path = Path(__file__).parent / "sxs_cache"
    sxs.write_config(cache_directory=str(local_cache_path.resolve()))
    print(f"\n[conftest.py] SXS cache configured at: {local_cache_path.resolve()}")

    try:
        # 'yield' passes control to the test session. All your tests will run now.
        yield
    finally:
        # --- Teardown Phase ---
        # This code is GUARANTEED to run after all tests are finished,
        # even if some of them fail.
        sxs.utilities.sxs_path_to_system_path = original_function
        print("\n[conftest.py] Restored original sxs path function.")