/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

namespace cuspa {

void bind_index_ops(nanobind::module_& m);
void bind_query_ops(nanobind::module_& m);
void bind_aggregate_ops(nanobind::module_& m);

}  // namespace cuspa
