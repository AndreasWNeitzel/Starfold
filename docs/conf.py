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
html_static_path = ["_static"]
templates_path = ["_templates"]
html_favicon = "_static/favicon.svg"
html_css_files = [
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;1,400"
    "&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400"
    "&family=IBM+Plex+Serif:wght@500;600&display=swap",
    "starfold.css",
]
html_js_files = ["starfold.js"]

_font_stacks = {
    "font-stack": (
        '"IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", '
        "Helvetica, Arial, sans-serif"
    ),
    "font-stack--headings": (
        '"IBM Plex Serif", Charter, "Bitstream Charter", Cambria, Georgia, serif'
    ),
    "font-stack--monospace": (
        '"IBM Plex Mono", "SFMono-Regular", Menlo, Consolas, "Liberation Mono", '
        "monospace"
    ),
}

html_theme_options = {
    "light_logo": "starfold-mark-light.svg",
    "dark_logo": "starfold-mark-dark.svg",
    "light_css_variables": {
        **_font_stacks,
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f3f6f8",
        "color-background-hover": "#e7edf1",
        "color-background-hover--transparent": "#e7edf100",
        "color-background-border": "#dfe5ea",
        "color-background-item": "#c9d2d9",
        "color-foreground-primary": "#1b1f24",
        "color-foreground-secondary": "#3d4650",
        "color-foreground-muted": "#5b6670",
        "color-foreground-border": "#b9c3cb",
        "color-brand-primary": "#0b6e7a",
        "color-brand-content": "#0b6e7a",
        "color-brand-visited": "#0b6e7a",
        "color-link-underline": "#a9cdd2",
        "color-link-underline--hover": "#0b6e7a",
        "color-link-underline--visited": "#a9cdd2",
        "color-highlighted-background": "#d5ebee",
        "color-highlight-on-target": "#f8eedb",
        "color-inline-code-background": "#f3f6f8",
        "color-admonition-background": "transparent",
        "color-card-border": "#dfe5ea",
        "color-card-background": "#ffffff",
        "color-card-marginals-background": "#f3f6f8",
        "color-table-header-background": "#f3f6f8",
        "color-table-border": "#dfe5ea",
        "color-api-name": "#1b1f24",
        "color-api-pre-name": "#5b6670",
        "color-api-background-hover": "#f3f6f8",
        "sf-accent": "#b7791f",
        "sf-code-background": "#f6f8fa",
        "sf-plate-border": "#dfe5ea",
    },
    "dark_css_variables": {
        **_font_stacks,
        "color-background-primary": "#0f1419",
        "color-background-secondary": "#161c23",
        "color-background-hover": "#1d252e",
        "color-background-hover--transparent": "#1d252e00",
        "color-background-border": "#263039",
        "color-background-item": "#3a4651",
        "color-foreground-primary": "#d7dde3",
        "color-foreground-secondary": "#b6bfc8",
        "color-foreground-muted": "#9aa5b1",
        "color-foreground-border": "#3a4651",
        "color-brand-primary": "#4fb3bf",
        "color-brand-content": "#4fb3bf",
        "color-brand-visited": "#4fb3bf",
        "color-link-underline": "#2d5d64",
        "color-link-underline--hover": "#4fb3bf",
        "color-link-underline--visited": "#2d5d64",
        "color-highlighted-background": "#12393f",
        "color-highlight-on-target": "#33291a",
        "color-inline-code-background": "#161c23",
        "color-admonition-background": "transparent",
        "color-card-border": "#263039",
        "color-card-background": "#0f1419",
        "color-card-marginals-background": "#161c23",
        "color-table-header-background": "#161c23",
        "color-table-border": "#263039",
        "color-api-name": "#d7dde3",
        "color-api-pre-name": "#9aa5b1",
        "color-api-background-hover": "#161c23",
        "sf-accent": "#e0b25a",
        "sf-code-background": "#131a20",
        "sf-plate-border": "#3a4651",
    },
}
html_show_sourcelink = False
pygments_style = "friendly"
pygments_dark_style = "github-dark"
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True
autosummary_generate = False
