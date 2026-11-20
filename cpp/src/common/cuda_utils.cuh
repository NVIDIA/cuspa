/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace cuspa {

// --- small CUDA helpers -----------------------------------------------------

#define CS_CHECK(expr)                                                                  \
  do {                                                                                  \
    auto _err = (expr);                                                                 \
    if (_err != cudaSuccess) {                                                          \
      (void)cudaGetLastError();                                                         \
      throw std::runtime_error(std::string{"CUDA error: "} + cudaGetErrorString(_err)); \
    }                                                                                   \
  } while (0)

void* allocate_scratch(std::size_t bytes, cudaStream_t stream);

struct DeviceScratch {
  void* ptr     = nullptr;
  std::size_t n = 0;
  cudaStream_t stream;

  explicit DeviceScratch(cudaStream_t s) : stream{s} {}
  ~DeviceScratch() {
    if (ptr)
      cudaFreeAsync(ptr, stream);
  }
  DeviceScratch(DeviceScratch const&)            = delete;
  DeviceScratch& operator=(DeviceScratch const&) = delete;

  void resize(std::size_t bytes) {
    if (bytes <= n)
      return;
    if (ptr) {
      CS_CHECK(cudaFreeAsync(ptr, stream));
      ptr = nullptr;
      n   = 0;
    }
    ptr = allocate_scratch(bytes, stream);
    n   = bytes;
  }
};

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  cudaStream_t stream;

  DeviceBuffer(std::size_t count, cudaStream_t s) : stream{s} {
    if (count)
      ptr = static_cast<T*>(allocate_scratch(sizeof(T) * count, stream));
  }
  ~DeviceBuffer() {
    if (ptr)
      cudaFreeAsync(ptr, stream);
  }
  DeviceBuffer(DeviceBuffer const&)            = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
};

}  // namespace cuspa
