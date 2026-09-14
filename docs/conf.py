# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "python"))

with (HERE.parent / "pyproject.toml").open("rb") as pyproject_file:
    package_metadata = tomllib.load(pyproject_file)["project"]

project = "cuspa"
author = "NVIDIA Corporation"
copyright = "2026, NVIDIA Corporation"
version = release = package_metadata["version"]

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_copybutton",
]
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"
autodoc_member_order = "bysource"
autodoc_mock_imports = ["cuspa._core", "cupy", "torch"]
default_role = "literal"
numpydoc_show_class_members = False
myst_enable_extensions = ["colon_fence", "deflist", "html_admonition"]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/cuspa",
            "icon": "fa-brands fa-github",
        }
    ],
    "show_nav_level": 2,
    "navigation_with_keys": False,
    # Disable external integrations while the documentation is private.
    "public_docs_features": False,
    "show_toc_level": 2,
}
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False
html_title = "cuspa"
