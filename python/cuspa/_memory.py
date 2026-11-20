# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import numpy as np


def _collect_host(chunks, dtype, tail=(), *, size=None, out=None):
    """Fill the output directly, growing a host buffer only for unknown row counts."""
    dtype = np.dtype(dtype)
    if out is None and size is not None:
        out = np.empty((size, *tail), dtype)
    if out is not None:
        shape = tuple(out.shape)
        if not shape or shape[1:] != tail or (size is not None and shape[0] != size):
            raise ValueError("out has the wrong shape")
        pos = 0
        for block in chunks:
            end = pos + len(block)
            if end > out.shape[0]:
                raise ValueError("out is too short for the points")
            out[pos:end] = block
            pos = end
        if pos != out.shape[0]:
            raise ValueError("out length differs from the number of points")
        return out

    buffer = bytearray()
    for block in chunks:
        if len(block):
            buffer.extend(memoryview(block).cast("B"))
    return np.frombuffer(buffer, dtype=dtype).reshape(-1, *tail)


def release_cached_blocks() -> None:
    """Return unused framework blocks before retrying a native scratch allocation."""
    if (cp := sys.modules.get("cupy")) is not None:
        default = cp.get_default_memory_pool()
        active = getattr(cp.cuda.get_allocator(), "__self__", None)
        if active is not default and hasattr(active, "free_all_blocks"):
            active.free_all_blocks()
        default.free_all_blocks()
    if (torch := sys.modules.get("torch")) is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
