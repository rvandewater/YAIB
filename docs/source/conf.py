import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Add your project root to Python path
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Yet Another ICU Benchmark'
copyright = '2025, Robin P. van de Water, Hendrik Schmidt, Patrick Rockenschaub, MIT License'
author = 'Robin P. van de Water, Hendrik Schmidt, Patrick Rockenschaub'
release = '1.0'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # For Google/NumPy style docstrings
    'sphinx_immaterial'
]
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme configuration
html_theme = 'sphinx_immaterial'
# Logo configuration
html_logo = '../figures/yaib_logo.png'  # Path to your logo file
html_favicon = '../figures/yaib_logo.png'  # Optional: favicon
# Theme options
# html_theme_options = {
#     'canonical_url': '',
#     'analytics_id': '',
#     'logo_only': False,
#     'display_version': True,
#     'prev_next_buttons_location': 'bottom',
#     'style_external_links': False,
#     'vcs_pageview_mode': '',
#     'style_nav_header_background': 'white',
#     # Toc options
#     'collapse_navigation': True,
#     'sticky_navigation': True,
#     'navigation_depth': 4,
#     'includehidden': True,
#     'titles_only': False,
#     'body_max_width': 'none',
#     'page_width': 'auto',
# }
# sphinx_immaterial theme options


html_static_path = ['_static']



# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Generate autosummary even if no references
autosummary_generate = True