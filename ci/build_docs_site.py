# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the versioned documentation site.

The site serves the checkout as ``latest/``, each published release as
``<version>/``, and the newest final release again as ``stable/``.
``switcher.json`` lists the versions for the version switcher.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from packaging.version import Version

BASE_URL = "https://nvidia.github.io/cuspa/"
REPO = Path(__file__).resolve().parent.parent
TAG_PATTERN = re.compile(r"v\d+\.\d+\.\d+(rc\d+)?")
REDIRECT = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Redirecting</title>
<meta http-equiv="refresh" content="0; url={target}">
<script>location.replace("{target}" + location.hash);</script>
</head>
<body><a href="{target}">{target}</a></body>
</html>
"""


def select_releases(tags: list[str]) -> list[Version]:
    """Return the releases to publish, newest first.

    Every final release is published. A release candidate is published only
    while it is newer than the newest final release.
    """
    releases = []
    for tag in tags:
        if TAG_PATTERN.fullmatch(tag):
            releases.append(Version(tag.removeprefix("v")))
        else:
            print(f"Skipping {tag}: not a release tag", file=sys.stderr)
    releases.sort(reverse=True)
    finals = [v for v in releases if not v.is_prerelease]
    return [v for v in releases if not v.is_prerelease or not finals or v > finals[0]]


def sphinx_build(python: Path, source: Path, output: Path, *, strict: bool) -> None:
    warnings_as_errors = ["-W"] if strict else []
    subprocess.run(
        [python, "-m", "sphinx", "-b", "html", *warnings_as_errors, "--keep-going"]
        + [source, output],
        check=True,
    )


def build_release(version: Version, output: Path, work: Path) -> None:
    tag = f"v{version}"
    tree = work / tag
    subprocess.run(
        ["git", "-C", REPO, "worktree", "add", "--detach", tree, tag], check=True
    )
    try:
        # Install the documentation dependencies as of the release, so that
        # newer Sphinx or theme releases cannot break the build.
        timestamp = subprocess.run(
            ["git", "-C", REPO, "log", "-1", "--format=%ct", tag],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        cutoff = datetime.fromtimestamp(int(timestamp), UTC)
        venv = work / f"{tag}-venv"
        python = venv / "bin" / "python"
        subprocess.run(["uv", "venv", "--python", sys.executable, venv], check=True)
        subprocess.run(
            ["uv", "pip", "install", "--python", python]
            + ["--exclude-newer", cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")]
            + ["--requirement", tree / "docs" / "requirements.txt"],
            check=True,
        )
        # A warning in an old release must not keep the site from deploying.
        sphinx_build(python, tree / "docs", output, strict=False)
    finally:
        subprocess.run(
            ["git", "-C", REPO, "worktree", "remove", "--force", tree], check=True
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for the site")
    parser.add_argument("tags", nargs="*", help="tags of the published releases")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True)
    releases = select_releases(args.tags)
    stable = next((v for v in releases if not v.is_prerelease), None)

    sphinx_build(Path(sys.executable), REPO / "docs", output / "latest", strict=True)
    with tempfile.TemporaryDirectory() as work:
        for version in releases:
            build_release(version, output / str(version), Path(work))
    if stable is not None:
        shutil.copytree(output / str(stable), output / "stable")

    switcher = [{"name": "latest", "version": "latest", "url": f"{BASE_URL}latest/"}]
    for version in releases:
        if version == stable:
            switcher.append(
                {
                    "name": f"{version} (stable)",
                    "version": str(version),
                    "url": f"{BASE_URL}stable/",
                    "preferred": True,
                }
            )
        else:
            switcher.append({"version": str(version), "url": f"{BASE_URL}{version}/"})
    (output / "switcher.json").write_text(json.dumps(switcher, indent=2) + "\n")

    # Before the site had versions, its pages were at the top level. Redirect
    # those addresses, and the site root, to the default version.
    default = output / ("latest" if stable is None else "stable")
    for page in default.rglob("*.html"):
        path = page.relative_to(default)
        if path.parts[0].startswith("_"):  # Sphinx assets, not pages
            continue
        stub = output / path
        stub.parent.mkdir(parents=True, exist_ok=True)
        target = page.relative_to(stub.parent, walk_up=True).as_posix()
        stub.write_text(REDIRECT.format(target=target))


if __name__ == "__main__":
    main()
