/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../ops.hpp"

namespace nb = nanobind;

namespace cuspa {
namespace {

template <typename T>
using Vector = nb::ndarray<nb::numpy, T, nb::device::cpu, nb::ndim<1>, nb::c_contig>;
using Input  = Vector<std::int32_t const>;

// Only occupied pages have storage. Each set bit gets a permanent CSR offset
// once nnz() has computed the prefix sums of the words in key order.
struct Page {
  static constexpr int words = 32;
  std::uint64_t bits[words]{};
  std::uint16_t prefix[words]{};
  std::int64_t start{};
};

class HostCSRPlan {
  std::int64_t cells_, genes_, size_ = -1;
  std::unordered_map<std::int64_t, Page> pages_;
  std::vector<std::int64_t> order_;
  std::vector<Page*> directory_;

  template <bool mark>
  void visit(Input labels, Input genes, std::int32_t* output, std::int32_t const* counts) {
    if (mark != (size_ < 0))
      throw std::logic_error("mark must precede nnz() and add must follow it");
    auto n = labels.shape(0);
    if (n != genes.shape(0))
      throw std::invalid_argument("labels and genes must have equal length");
    // A direct page lookup costs at most 16 pointers per occupied page.
    // Map values keep their addresses when newly occupied pages trigger a rehash.
    constexpr std::size_t page_bits = 64 * Page::words;
    auto possible = (static_cast<std::uint64_t>(cells_ * genes_) + page_bits - 1) / page_bits;
    if (directory_.empty() && possible && (possible - 1) / 16 < pages_.size()) {
      directory_.resize(possible);
      for (auto& [id, page] : pages_)
        directory_[id] = &page;
    }
    auto const* rows    = labels.data();
    auto const* columns = genes.data();
    std::int64_t last   = -1;
    Page* page          = nullptr;
    for (std::size_t i = 0; i < n; ++i) {
      auto row = rows[i], col = columns[i];
      if (col >= genes_)
        throw std::invalid_argument("gene_ids contains a value outside num_genes");
      if (row < -1 || row >= cells_)
        throw std::invalid_argument("cell ID outside matrix shape");
      if (row == -1 || col < 0)
        continue;
      auto key = row * genes_ + col, id = key / (64 * Page::words);
      if (id != last) {
        page = directory_.empty() ? nullptr : directory_[id];
        if (!page) {
          if constexpr (mark) {
            page = &pages_[id];
            if (!directory_.empty())
              directory_[id] = page;
          } else {
            page = &pages_.at(id);
          }
        }
        last = id;
      }
      auto word = (key / 64) % Page::words;
      auto mask = std::uint64_t{1} << (key % 64);
      if constexpr (mark) {
        page->bits[word] |= mask;
      } else {
        auto bits = page->bits[word];
        if (!(bits & mask))
          throw std::out_of_range("cell/gene pair missing from first pass");
        auto& value =
            output[page->start + page->prefix[word] + __builtin_popcountll(bits & (mask - 1))];
        auto increment = counts ? counts[i] : 1;
        if (increment < 0 || value < 0)
          throw std::invalid_argument("counts must be non-negative");
        if (value > std::numeric_limits<std::int32_t>::max() - increment)
          throw std::overflow_error("cell/gene count exceeds int32");
        value += increment;
      }
    }
  }

public:
  HostCSRPlan(std::int64_t cells, std::int64_t genes) : cells_(cells), genes_(genes) {
    if (cells < 0 || genes < 0)
      throw std::invalid_argument("matrix dimensions must be non-negative");
    if (cells && genes > std::numeric_limits<std::int64_t>::max() / cells)
      throw std::overflow_error("matrix keys exceed int64");
  }
  void mark(Input labels, Input genes) { visit<true>(labels, genes, nullptr, nullptr); }
  std::int64_t nnz() {
    if (size_ >= 0)
      return size_;
    order_.reserve(pages_.size());
    for (auto const& item : pages_)
      order_.push_back(item.first);
    std::sort(order_.begin(), order_.end());
    size_ = 0;
    for (auto id : order_) {
      auto& page = pages_.at(id);
      page.start = size_;
      for (int w = 0; w < Page::words; ++w) {
        page.prefix[w] = static_cast<std::uint16_t>(size_ - page.start);
        size_ += __builtin_popcountll(page.bits[w]);
      }
    }
    return size_;
  }
  template <typename I>
  void layout(Vector<I> indptr, Vector<I> indices) {
    if (size_ < 0 || indptr.size() != static_cast<std::size_t>(cells_) + 1 ||
        indices.size() != static_cast<std::size_t>(size_))
      throw std::invalid_argument("layout needs buffers sized from nnz() and matrix shape");
    if (size_ > std::numeric_limits<I>::max())
      throw std::overflow_error("CSR offsets need int64");
    std::int64_t row = 0, base = 0, pos = 0;
    auto* offsets = indptr.data();
    auto* cols    = indices.data();
    offsets[0]    = 0;
    for (auto id : order_) {
      auto const& page = pages_.at(id);
      for (int w = 0; w < Page::words; ++w) {
        for (auto bits = page.bits[w]; bits; bits &= bits - 1) {
          auto key = (id * Page::words + w) * 64 + __builtin_ctzll(bits);
          while (key - base >= genes_) {
            offsets[++row] = static_cast<I>(pos);
            base += genes_;
          }
          cols[pos++] = static_cast<I>(key - base);
        }
      }
    }
    std::fill(offsets + row + 1, offsets + indptr.size(), static_cast<I>(pos));
  }
  void add(Input labels, Input genes, Vector<std::int32_t> output, std::optional<Input> counts) {
    if (size_ < 0 || output.size() != static_cast<std::size_t>(size_) ||
        (counts && counts->size() != labels.size()))
      throw std::invalid_argument("output must have nnz() entries and counts must match labels");
    visit<false>(labels, genes, output.data(), counts ? counts->data() : nullptr);
  }
};

}  // namespace

void bind_host_plan(nb::module_& m) {
  using namespace nb::literals;
  nb::class_<HostCSRPlan>(m, "HostCSRPlan")
      .def(nb::init<std::int64_t, std::int64_t>(), "cells"_a, "genes"_a)
      .def("mark", &HostCSRPlan::mark, "labels"_a.noconvert(), "genes"_a.noconvert())
      .def("nnz", &HostCSRPlan::nnz)
      .def("layout",
           &HostCSRPlan::layout<std::int32_t>,
           "out_indptr"_a.noconvert(),
           "out_indices"_a.noconvert())
      .def("layout",
           &HostCSRPlan::layout<std::int64_t>,
           "out_indptr"_a.noconvert(),
           "out_indices"_a.noconvert())
      .def("add",
           &HostCSRPlan::add,
           "labels"_a.noconvert(),
           "genes"_a.noconvert(),
           "out_data"_a.noconvert(),
           "counts"_a.noconvert() = nb::none());
}
}  // namespace cuspa
