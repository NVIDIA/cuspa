# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from cuspa import _core


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_host_csr_plans_sparse_wide_rows_and_exact_counts(dtype):
    genes = 2**31 - 1
    plan = _core.HostCSRPlan(4, genes)
    cells = np.array([2, 0, -1, 2, 2, 2, 0], np.int32)
    ids = np.array([genes - 1, 7, 3, 0, genes - 1, -1, 7], np.int32)
    plan.mark(cells[:4], ids[:4])
    plan.mark(cells[4:], ids[4:])
    assert plan.nnz() == 3
    ptr, cols = np.empty(5, dtype), np.empty(3, dtype)
    plan.layout(ptr, cols)
    np.testing.assert_array_equal(ptr, [0, 1, 1, 3, 3])
    np.testing.assert_array_equal(cols, [7, 0, genes - 1])
    counts = np.zeros(3, np.int32)
    plan.add(cells, ids, counts)
    np.testing.assert_array_equal(counts, [2, 1, 2])
    counts[:] = np.iinfo(np.int32).max
    with pytest.raises(OverflowError):
        plan.add(cells, ids, counts)
