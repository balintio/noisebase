# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Noisebase'
copyright = '2024, Martin Bálint'
author = 'Martin Bálint'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'env']
extensions = [
    'myst_parser', 
    'sphinx_copybutton', 
    'sphinx.ext.napoleon', 
    'sphinx.ext.autodoc',
    'sphinx_codeautolink'
]
napoleon_custom_sections = [('Returns', 'params_style')]
myst_enable_extensions = [
    'attrs_block',
    'colon_fence'
]
autoclass_content = 'both'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_logo = '_static/logo_small-01.png'
html_theme_options = {
    "sidebar_hide_name": True,
}
html_favicon = '_static/favicon.ico'
#html_theme = 'classic'
html_static_path = ['_static']
