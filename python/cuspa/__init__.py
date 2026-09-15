# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU spatial operations for spatial omics."""

from __future__ import annotations

from . import io, tl
from ._types import Polygons, SpatialIndex

try:
    from ._version import __version__
except ModuleNotFoundError:
    __version__ = "0+unknown"

__all__ = ["Polygons", "SpatialIndex", "io", "tl"]
