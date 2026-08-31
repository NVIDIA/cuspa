# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate that a GPU CI job received its requested accelerator."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def validate_device(
    properties: Mapping[str, object],
    compute_capability: object,
    *,
    name_contains: str,
    expected_compute_capability: str,
) -> tuple[str, str]:
    """Validate and return the normalized CUDA device name and capability."""
    raw_name = properties["name"]
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    actual = str(compute_capability).replace(".", "")
    expected = expected_compute_capability.replace(".", "")

    if name_contains.casefold() not in name.casefold():
        raise ValueError(
            f"expected GPU name containing {name_contains!r}, found {name!r}"
        )
    if actual != expected:
        raise ValueError(
            f"expected compute capability {expected_compute_capability}, "
            f"found {compute_capability}"
        )
    return name, actual


def main() -> None:
    """Check the active CuPy device against the requested CI target."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-contains", required=True)
    parser.add_argument("--compute-capability", required=True)
    args = parser.parse_args()

    import cupy as cp

    name, capability = validate_device(
        cp.cuda.runtime.getDeviceProperties(0),
        cp.cuda.Device(0).compute_capability,
        name_contains=args.name_contains,
        expected_compute_capability=args.compute_capability,
    )
    print(f"validated GPU: {name} (compute capability {capability})")


if __name__ == "__main__":
    main()
