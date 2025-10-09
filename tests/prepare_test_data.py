# Prepares data for compatibility purposes 

import sxs
from pathlib import Path

print("--- Preparing SXS test data for cross-platform compatibility ---")


original_sxs_path_converter = sxs.utilities.sxs_path_to_system_path


def universal_safe_sxs_path_converter(path):

    original_path = original_sxs_path_converter(path)

    return str(original_path).replace(":", "-").replace("_", "-")


sxs.utilities.sxs_path_to_system_path = universal_safe_sxs_path_converter
print("✅ sxs.utilities.sxs_path_to_system_path has been patched.")


# Place cache in local repository
local_cache_path = Path(__file__).parent / "sxs_cache"
local_cache_path.mkdir(parents=True, exist_ok=True)
sxs.write_config(cache_directory=str(local_cache_path.resolve()))
print(f"✅ SXS cache configured at: {local_cache_path.resolve()}")


print("\nDownloading test simulations...")

simulations_to_download = [
    "SXS:BBH:2325v3.0/Lev3/metadata.json",
    "SXS:BBH:2325v3.0/Lev3:Strain_N2.h5",
    "SXS:BBH:2325v3.0/Lev3:Strain_N2.json",
    "SXS:BBH:2325v3.0/Lev3:ExtraWaveforms.h5",
    "SXS:BBH:2325v3.0/Lev3:ExtraWaveforms.json",
]

for sim in simulations_to_download:
    print(f"  -> Loading {sim}...")
    try:
        sxs.load(sim, download=True)
    except Exception as e:
        print(f"    ERROR: Could not load {sim}. Reason: {e}")

print("\n--- ✅ Test data preparation complete! ---")
print("You can now commit the 'tests/sxs_cache' directory.")