# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Set the distribution name and GPU targets for a CUDA wheel build."""

from __future__ import annotations

import argparse
from pathlib import Path

import tomlkit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-major", required=True, choices=("12", "13"))
    parser.add_argument("--architectures", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()

    document = tomlkit.parse(args.pyproject.read_text(encoding="utf-8"))
    project = document["project"]
    project["name"] = f"cuspa-cu{args.cuda_major}"
    dynamic = project.get("dynamic")
    if dynamic is None or "version" not in dynamic:
        raise ValueError("project.version must be dynamic")
    dynamic.remove("version")
    if dynamic:
        project["dynamic"] = dynamic
    else:
        del project["dynamic"]
    project["version"] = args.version
    document["tool"]["scikit-build"]["cmake"]["define"]["CMAKE_CUDA_ARCHITECTURES"] = (
        args.architectures
    )
    args.pyproject.write_text(tomlkit.dumps(document), encoding="utf-8")


if __name__ == "__main__":
    main()
