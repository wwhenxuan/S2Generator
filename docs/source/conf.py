# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("agg")
matplotlib.rcParams["savefig.dpi"] = 100
matplotlib.rcParams["figure.max_open_warning"] = 40

# -- Project information -----------------------------------------------------

project = "S2Generator"
copyright = f"2025-{date.today().year}, the S2Generator team"
author = "the S2Generator team"

sys.path.insert(0, str(Path("..", "..").resolve()))

import s2generator

release = s2generator.__version__
version = s2generator.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_gitstamp",
    "sphinx_gallery.gen_gallery",
]

autosectionlabel_prefix_document = True

# Subsection paths are relative to this file (docs/source/conf.py).
_GALLERY_SECTION_ORDER = [
    "../../examples/symbol",
    "../../examples/excitation",
    "../../examples/simulator",
    "../../examples/scm",
    "../../examples/augmentation",
    "../../examples/tools",
]

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py$",
    "ignore_pattern": r"(main\.py|convert_ipynb_to_py\.py|__init__\.py)",
    "subsection_order": _GALLERY_SECTION_ORDER,
    "nested_sections": True,
    "download_all_examples": True,
    "reset_modules": ("matplotlib",),
    "abort_on_example_error": False,
    "min_reported_time": 1,
    "matplotlib_animations": False,
}

autodoc_default_options = {
    "members": True,
    "imported-members": True,
    "show-inheritance": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
}

templates_path = []

source_suffix = {".rst": "restructuredtext"}

html_logo = "_static/S2Generator_logo.png"
html_favicon = "_static/S2Generator_logo.png"

language = "en"

show_warning_types = True
suppress_warnings = []

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
mathjax3_config = {
    "tex": {
        "processEscapes": True,
    }
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_css_files = ["theme_overrides.css", "custom.css"]

# TODO (author): point these at the live Read the Docs project once it exists.
_switcher_json = (
    "https://s2generator.readthedocs.io/en/latest/_static/version_switcher.json"
)
_version_match = os.environ.get("READTHEDOCS_VERSION") or "latest"

html_theme_options = {
    "announcement": "",
    "logo": {
        "image_light": "_static/S2Generator_logo.png",
        "image_dark": "_static/S2Generator_logo.png",
        "text": "S2Generator",
        "alt_text": "S2Generator",
        # TODO (author): replace with the live Read the Docs URL.
        "link": "https://s2generator.readthedocs.io/",
    },
    "switcher": {
        "json_url": _switcher_json,
        "version_match": _version_match,
    },
    "check_switcher": False,
    "show_version_warning_banner": True,
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wwhenxuan/S2Generator",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/s2generator/",
            "icon": "https://raw.githubusercontent.com/wwhenxuan/S2Generator/main/docs/source/_static/logo-pypi.svg",
            "type": "url",
        },
    ],
    "navbar_align": "content",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": {
        "**": ["page-toc", "sourcelink"],
    },
    "show_toc_level": 4,
    "collapse_navigation": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "pygments_light_style": "xcode",
    "pygments_dark_style": "monokai",
    "show_prev_next": False,
    "show_nav_level": 1,
    "back_to_top_button": True,
}

remove_from_toctrees = []

html_sidebars = {
    "index": [],
    "API/*": [],
    "auto_examples/index": [],
    "auto_examples/*/index": [],
    "release_notes/index": [],
}
html_show_sourcelink = False

htmlhelp_basename = "s2generatordoc"
