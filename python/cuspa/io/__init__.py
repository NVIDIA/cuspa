# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional adapters for external data sources.

Each adapter imports its heavyweight dependency (spatialdata, shapely, …)
lazily so the core package stays import-safe with no extras installed.
Install what you need with e.g. ``pip install cuspa[spatialdata]``.
"""

from __future__ import annotations

from ._geoarrow import from_geoarrow, from_geopandas
from ._spatialdata import Transcripts, from_spatialdata, transcripts_from_spatialdata

__all__ = [
    "Transcripts",
    "from_geoarrow",
    "from_geopandas",
    "from_spatialdata",
    "transcripts_from_spatialdata",
]
