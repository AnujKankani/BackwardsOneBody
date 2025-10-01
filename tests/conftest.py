# Patches SXS functions for testing
import pytest
import sxs
from pathlib import Path

@pytest.fixture(scope="session", autouse=True)
def sxs_test_environment_setup():

    original_function = sxs.utilities.sxs_path_to_system_path


    def universal_safe_sxs_path_converter(path):
        """A patched version that replaces colons with hyphens, regardless of OS."""
        raw_path = original_function(path)
        return str(raw_path).replace(":", "-").replace("_", "-")


    sxs.utilities.sxs_path_to_system_path = universal_safe_sxs_path_converter

    local_cache_path = Path(__file__).parent / "sxs_cache"
    sxs.write_config(cache_directory=str(local_cache_path.resolve()))
    print(f"\n[conftest.py] SXS cache configured at: {local_cache_path.resolve()}")

    try:
       
        yield
    finally:

        sxs.utilities.sxs_path_to_system_path = original_function
        print("\n[conftest.py] Restored original sxs path function.")