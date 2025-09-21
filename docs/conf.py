# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import date
import os
import sys
from pathlib import Path

project = "S2Generator"
copyright = f"2025-{date.today().year}, the S2Generator team"
author = "the S2Generator team"

release = "0.0.2"
version = "0.0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.append(str(Path("source").resolve()))
sys.path.insert(0, str(Path("..").resolve()))
sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.napoleon",
    "myst_parser",
]

apidoc_modules = [
    {
        "path": "../S2Generator",
        "destination": "source/API",
        "exclude_patterns": ["**/test*"],
        "max_depth": 4,
        "follow_links": False,
        "separate_modules": False,
        "include_private": True,
        "no_headings": False,
        "module_first": False,
        "implicit_namespaces": True,
        "automodule_options": {"members", "show-inheritance", "undoc-members"},
    },
]

templates_path = ["_templates"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown", ".txt": "markdown"}

html_logo = "source/_static/S2Generator_logo.png"
html_favicon = "source/_static/S2Generator_logo.png"

language = "en"

show_warning_types = False
suppress_warnings = []

sphinx_gallery_conf = {
    "examples_dirs": "../examples/_py",
    "gallery_dirs": "auto_examples",
    "doc_module": ("S2Generator",),
    "within_subsection_order": "FileNameSortKey",
    "plot_gallery": True,
    "run_stale_examples": False,
    "remove_config_comments": True,
    "min_reported_time": 0.5,
    "run_stale_examples": True,
    "show_memory": False,
    "filename_pattern": ".*",
    "expected_failing_examples": [],
    "recommender": {"enable": True, "n_examples": 5, "min_df": 3, "max_df": 0.9},
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
mathjax3_config = {
    "tex": {
        "processEscapes": True,
    }
}
myst_enable_extensions = [
    "amsmath",      # 支持 LaTeX 的 amsmath 环境，如 \begin{align}
    "dollarmath",   # 启用 $...$ 和 $$...$$ 语法
]


duration_show_files = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["source/_static"]

html_css_files = ["theme_overrides.css", "custom.css"]

html_theme_options = {
    "announcement": "",  # You can specify an arbitrary URL that will be used as the HTML source for your announcement.
    # Navigation bar
    "logo": {
        "text": "S2Generator",
        "link": "https://S2Generator.readthedocs.io/",
    },
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wwhenxuan/S2Generator",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/S2Generator/",
            "icon": "https://raw.githubusercontent.com/changewam/PySDKit/refs/heads/main/docs/source/_static/logo-pypi.svg",
            "type": "url",
        },
    ],
    "navbar_align": "content",
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    # "navbar_end": ["navbar-icon-links"],
    # -----------------------------------------------------------------------------
    "switcher": {
        "json_url": (
            "https://raw.githubusercontent.com/changewam/S2Generator/refs/heads/main/docs/source/_static/version_switcher.json"
        ),  # the persistent location of the JSON file
        "version_match": "dev" if "dev" in version else version,
    },
    "show_version_warning_banner": True,
    # Secondary_sidebar
    "secondary_sidebar_items": {
        "**": ["page-toc", "sourcelink"],
        "examples/*": [
            "page-toc",
            "sourcelink",
            "sg_execution_times",
            "sg_download_links",
            "sg_launcher_links",
        ],
        "index": [],
    },
    "show_toc_level": 3,
    "collapse_navigation": True,
    # Footer
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    # Color
    "pygments_light_style": "xcode",
    "pygments_dark_style": "monokai",
    # Other
    "show_prev_next": False,
    "show_nav_level": 2,
    "back_to_top_button": True,
    "use_edit_page_button": False,
}

remove_from_toctrees = []

# Custom sidebar templates, maps document names to template names.
html_sidebars = {"index": []}  # Hide sidebar in home page
html_show_sourcelink = False
