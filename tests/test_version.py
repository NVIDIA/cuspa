# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import cuspa


def test_runtime_version_matches_distribution() -> None:
    installed_versions = set()
    for distribution_name in ("cuspa", "cuspa-cu12", "cuspa-cu13"):
        try:
            installed_versions.add(version(distribution_name))
        except PackageNotFoundError:
            pass

    assert installed_versions
    assert cuspa.__version__ in installed_versions
