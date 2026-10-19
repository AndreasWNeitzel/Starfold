"""Sphinx configuration for the Starfold manual."""

project = "Starfold"
author = "Andreas W. Neitzel"
copyright = "2026, Andreas W. Neitzel"  # noqa: A001
release = "0.2.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_design",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "data", "figures/v0_1", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
}

html_theme = "furo"
html_title = "Starfold manual"
html_static_path = []
html_show_sourcelink = False
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True
autosummary_generate = False
