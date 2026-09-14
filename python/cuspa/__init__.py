# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU spatial operations for spatial omics."""

from __future__ import annotations

from . import io, tl
from ._types import Polygons, SpatialIndex

__all__ = ["Polygons", "SpatialIndex", "io", "tl"]
__version__ = "0.0.1"
