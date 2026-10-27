# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from docutils.parsers.rst import roles
from packaging.version import Version

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "python"))

project = "cuspa"
author = "NVIDIA Corporation"
copyright = "2026, NVIDIA Corporation"
release = subprocess.run(
    [sys.executable, "-m", "hatchling", "version"],
    cwd=HERE.parent,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
version = release
# The site serves `main` as `latest` and each release under its version.
_version = Version(release)
if _version.is_devrelease or _version.local:
    docs_version = "latest"
else:
    docs_version = _version.public

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
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
autosummary_generate = True
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

extlinks = {
    "issue": ("https://github.com/NVIDIA/cuspa/issues/%s", "issue%s"),
    "pr": ("https://github.com/NVIDIA/cuspa/pull/%s", "pr%s"),
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # release-notes/index.md includes the notes of each version.
    "release-notes/[0-9]*.md",
    "release-notes/blank.md",
]

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
    "switcher": {
        "json_url": "https://nvidia.github.io/cuspa/switcher.json",
        "version_match": docs_version,
    },
    # The Docs workflow writes switcher.json when it deploys the site.
    "check_switcher": False,
    "show_version_warning_banner": True,
}
html_static_path = ["_static"]
html_css_files = ["css/override.css"]
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False
html_title = "cuspa"


def setup(app):
    # Release dates and authors in the release notes.
    for name in ("small", "smaller"):
        role = roles.CustomRole(name, roles.generic_custom_role, {"classes": [name]})
        app.add_role(name, role)
