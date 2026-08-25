# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information --
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OpenDP Cookbook'
copyright = '2026, OpenDP Project'
author = 'OpenDP Project'
release = '0.15.1' # Match the library version.


# -- General configuration --
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['nbsphinx', 'myst_parser']
templates_path = ['_templates']
exclude_patterns = ['conf.py'] # Other .py files WILL BE converted to notebooks and executed. 
nitpicky = True


# -- Options for HTML output --
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_logo = "_static/images/opendp-logo.png"
html_favicon = "favicon.ico"
html_css_files = ["css/custom.css"]
html_theme_options = {
    "github_url": "https://github.com/opendp/opendp-cookbook",
    "announcement": "https://raw.githubusercontent.com/opendp/opendp/refs/heads/main/docs/source/announcement.html",
}


# -- Options for nbsphinx output --
# https://nbsphinx.readthedocs.io/en/0.9.8/configuration.html

nbsphinx_custom_formats = {
    # TODO: Use dp-wizard-templates for more control.
    '.py': ['jupytext.reads', {'fmt': 'py:light'}],
}

# -- Options for Myst-Parser --
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html

myst_enable_extensions = [
    "gfm_autolink"
]