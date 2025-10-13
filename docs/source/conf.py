# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # for ReadTheDocs (root)
sys.path.insert(0, os.path.abspath('../'))
project = 'gwBOB'
copyright = '2025, Anuj Kankani'
author = 'Anuj Kankani'
release = '0.1.0'
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
source_suffix = [".rst", ".md"]
autosummary_generate = True

import inspect
import numpy as np




# =============================================================================
# Custom hook to forcefully shorten the t_shift_range NumPy array signature
# Place this entire block at the END of your conf.py file.
# =============================================================================
import re

def replace_long_numpy_repr(app, what, name, obj, options, signature, return_annotation):
    """
    This function finds the long array representation and replaces it with a short one.
    """
    if signature:
        # --- DEBUGGING: Check if this function is running ---
        # You should see this message in your terminal during the build.
        print(f"Processing signature for: {name}")

        # Define the string we want to find. We only need the start of it.
        offending_string_start = "t_shift_range=array"
        
        if offending_string_start in signature:
            # This is the short, clean string we want in our documentation
            replacement = "t_shift_range=np.arange(-10, 10, 0.1)"
            
            # We use a regular expression to find and replace the entire array definition,
            # from "array([...])" to our clean string.
            # This pattern is robust and handles the content inside the array.
            pattern = re.compile(r"t_shift_range=array\(\[.*?\]\s*\)", re.DOTALL)
            
            new_signature = pattern.sub(replacement, signature)

            # --- DEBUGGING: Check if the replacement worked ---
            print(f"  -> Found and replaced t_shift_range. New signature: {new_signature}")
            
            return new_signature, return_annotation

    # If we didn't find the parameter, return the original signature
    return signature, return_annotation

def setup(app):
    """
    Connects our custom function to the Sphinx build process.
    """
    # The 'priority' is important. A high number ensures this runs *after*
    # other extensions (like sphinx-autodoc-typehints) have done their processing.
    app.connect('autodoc-process-signature', replace_long_numpy_repr, priority=1500)