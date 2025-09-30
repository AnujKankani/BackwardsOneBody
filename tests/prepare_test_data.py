# prepare_test_data.py (Corrected Version)

import sxs
from pathlib import Path

print("--- Preparing SXS test data for cross-platform compatibility ---")

# --- Step 1: Save a reference to the original library function BEFORE we patch it ---
original_sxs_path_converter = sxs.utilities.sxs_path_to_system_path

# --- Step 2: Define our new, safe path converter ---
# This function calls the *original* function we saved, avoiding recursion.
def universal_safe_sxs_path_converter(path):
    """A patched version that replaces colons with hyphens, regardless of OS."""
    # Call the ORIGINAL function that we saved earlier.
    original_path = original_sxs_path_converter(path)
    # Sanitize its output to be safe on all systems.
    return str(original_path).replace(":", "-").replace("_", "-")

# --- Step 3: Monkeypatch the library ---
# Now we replace the library's function with our new, non-recursive one.
sxs.utilities.sxs_path_to_system_path = universal_safe_sxs_path_converter
print("✅ sxs.utilities.sxs_path_to_system_path has been patched.")


# --- Step 4: Configure the cache to be local to the repository ---
local_cache_path = Path(__file__).parent / "sxs_cache"
local_cache_path.mkdir(parents=True, exist_ok=True)
sxs.write_config(cache_directory=str(local_cache_path.resolve()))
print(f"✅ SXS cache configured at: {local_cache_path.resolve()}")


# --- Step 5: Download all necessary data for the tests ---
print("\nDownloading test simulations...")

# ---- List of simulations to download ----
simulations_to_download = [
    # Files for SXS:BBH:2325
    "SXS:BBH:2325v3.0/Lev3/metadata.json",
    "SXS:BBH:2325v3.0/Lev3:Strain_N2.h5",
    "SXS:BBH:2325v3.0/Lev3:Strain_N2.json",
    "SXS:BBH:2325v3.0/Lev3:ExtraWaveforms.h5",
    "SXS:BBH:2325v3.0/Lev3:ExtraWaveforms.json",# <-- Added JSON
    # Add any other simulations or specific files you need for tests here.
]
# -----------------------------------------

for sim in simulations_to_download:
    print(f"  -> Loading {sim}...")
    try:
        sxs.load(sim, download=True)
    except Exception as e:
        print(f"    ERROR: Could not load {sim}. Reason: {e}")

print("\n--- ✅ Test data preparation complete! ---")
print("You can now commit the 'tests/sxs_cache' directory.")