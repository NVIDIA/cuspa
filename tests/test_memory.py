# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from cuspa._memory import release_cached_blocks


def test_release_cached_blocks_preserves_live_allocations(monkeypatch):
    cp = pytest.importorskip("cupy")
    active = cp.cuda.MemoryPool()
    default = cp.cuda.MemoryPool()
    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: default)
    with cp.cuda.using_allocator(default.malloc):
        cached = cp.empty(512, cp.uint8)
        del cached
    with cp.cuda.using_allocator(active.malloc):
        live = cp.ones(128, cp.uint8)
        cached = cp.empty(1024, cp.uint8)
        del cached
        assert active.free_bytes() and default.free_bytes()
        release_cached_blocks()
        assert active.free_bytes() == default.free_bytes() == 0
        assert live.sum().item() == 128
