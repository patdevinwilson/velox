/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/cudf/CudfDefaultStreamOverload.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>

#include <common/base/Exceptions.h>
#include "velox/common/process/StackTrace.h"

#include <glog/logging.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string_view>

namespace facebook::velox::cudf_velox {

namespace {

char const*& allocLabel() {
  static thread_local char const* label = "unlabeled";
  return label;
}

// Device-backed pools are self-limiting, but a pool over managed memory keeps
// growing into host RAM once the device is full. Unbounded, several workers on
// one host will collectively exhaust it and get OOM-killed by the kernel (the
// coordinator is usually the casualty). Cap each pool at a percentage of total
// device memory; above 100% is deliberate host oversubscription.
std::optional<std::size_t> managedPoolMaxBytes(int maxPercent) {
  if (maxPercent <= 0) {
    return std::nullopt; // Explicitly unbounded.
  }
  std::size_t free = 0;
  std::size_t total = 0;
  auto const status = cudaMemGetInfo(&free, &total);
  VELOX_CHECK(
      status == cudaSuccess,
      "Failed to query device memory for managed pool sizing: {}",
      cudaGetErrorString(status));
  auto const bytes =
      static_cast<std::size_t>(static_cast<double>(total) * maxPercent / 100.0);
  // pool_memory_resource requires 256B alignment.
  return bytes & ~std::size_t{255};
}

// A single outsized allocation is what actually exhausts a pool, but RMM only
// reports the size, leaving the call site unknown. Wrapping the outermost
// resource lets us attribute such requests to a stack trace.
class BigAllocationTracerImpl {
 public:
  BigAllocationTracerImpl(
      cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
      std::size_t thresholdBytes)
      : upstream_{std::move(upstream)}, threshold_{thresholdBytes} {}

  BigAllocationTracerImpl(BigAllocationTracerImpl const&) = delete;
  BigAllocationTracerImpl(BigAllocationTracerImpl&&) = delete;
  BigAllocationTracerImpl& operator=(BigAllocationTracerImpl const&) = delete;
  BigAllocationTracerImpl& operator=(BigAllocationTracerImpl&&) = delete;

  bool operator==(BigAllocationTracerImpl const& other) const noexcept {
    return this == std::addressof(other);
  }

  bool operator!=(BigAllocationTracerImpl const& other) const noexcept {
    return !(*this == other);
  }

  void* allocate(
      cuda::stream_ref stream,
      std::size_t bytes,
      std::size_t alignment = alignof(std::max_align_t)) {
    maybeLog(bytes);
    return upstream_.allocate(stream, bytes, alignment);
  }

  void deallocate(
      cuda::stream_ref stream,
      void* ptr,
      std::size_t bytes,
      std::size_t alignment = alignof(std::max_align_t)) noexcept {
    upstream_.deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(
      std::size_t bytes,
      std::size_t alignment = alignof(std::max_align_t)) {
    maybeLog(bytes);
    return upstream_.allocate_sync(bytes, alignment);
  }

  void deallocate_sync(
      void* ptr,
      std::size_t bytes,
      std::size_t alignment = alignof(std::max_align_t)) noexcept {
    upstream_.deallocate_sync(ptr, bytes, alignment);
  }

  friend void get_property(
      BigAllocationTracerImpl const&,
      cuda::mr::device_accessible) noexcept {}

 private:
  void maybeLog(std::size_t bytes) const {
    if (bytes >= threshold_) {
      LOG(WARNING) << "[rmm:big] allocating " << bytes << " bytes ("
                   << static_cast<double>(bytes) /
              static_cast<double>(std::size_t{1} << 30)
                   << " GiB) site=" << allocLabel() << "\n"
                   << process::StackTrace().toString();
    }
  }

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  std::size_t threshold_;
};

class BigAllocationTracer
    : public cuda::mr::shared_resource<BigAllocationTracerImpl> {
  using shared_base = cuda::mr::shared_resource<BigAllocationTracerImpl>;

 public:
  BigAllocationTracer(
      cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
      std::size_t thresholdBytes)
      : shared_base(
            cuda::mr::make_shared_resource<BigAllocationTracerImpl>(
                std::move(upstream),
                thresholdBytes)) {}

  friend void get_property(
      BigAllocationTracer const&,
      cuda::mr::device_accessible) noexcept {}
};

static_assert(
    cuda::mr::resource_with<BigAllocationTracer, cuda::mr::device_accessible>,
    "BigAllocationTracer does not satisfy the cuda::mr::resource concept");

// Threshold in MiB for logging a stack trace per allocation; 0 disables.
std::size_t bigAllocationLogBytes() {
  char const* env = std::getenv("VELOX_CUDF_BIG_ALLOC_LOG_MB");
  if (env == nullptr) {
    return 0;
  }
  char* end = nullptr;
  auto const mb = std::strtoull(env, &end, 10);
  if (end == env) {
    return 0;
  }
  return static_cast<std::size_t>(mb) << 20;
}

cuda::mr::any_resource<cuda::mr::device_accessible> createBaseMemoryResource(
    std::string_view mode,
    int percent,
    int maxPercent) {
  if (mode == "cuda") {
    return rmm::mr::cuda_memory_resource{};
  } else if (mode == "pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "async") {
    return rmm::mr::cuda_async_memory_resource{};
  } else if (mode == "arena") {
    return rmm::mr::arena_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed") {
    return rmm::mr::managed_memory_resource{};
  } else if (mode == "managed_pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::managed_memory_resource{},
        rmm::percent_of_free_device_memory(percent),
        managedPoolMaxBytes(maxPercent));
  } else if (mode == "managed_async") {
    return rmm::mr::cuda_async_managed_memory_resource{};
  } else if (mode == "prefetch_managed") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::managed_memory_resource{});
  } else if (mode == "prefetch_managed_pool") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::pool_memory_resource(
            rmm::mr::managed_memory_resource{},
            rmm::percent_of_free_device_memory(percent),
            managedPoolMaxBytes(maxPercent)));
  } else if (mode == "prefetch_managed_async") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::cuda_async_managed_memory_resource{});
  }
  VELOX_FAIL(
      "Unknown memory resource mode: " + std::string(mode) +
      "\nExpecting: cuda, pool, async, arena, managed, prefetch_managed, " +
      "managed_pool, prefetch_managed_pool, managed_async, prefetch_managed_async");
}

} // namespace

cuda::mr::any_resource<cuda::mr::device_accessible> createMemoryResource(
    std::string_view mode,
    int percent,
    int maxPercent) {
  auto base = createBaseMemoryResource(mode, percent, maxPercent);
  auto const threshold = bigAllocationLogBytes();
  if (threshold == 0) {
    return base;
  }
  LOG(WARNING) << "[rmm:big] logging allocations >= " << (threshold >> 20)
               << " MiB with stack traces";
  return BigAllocationTracer{std::move(base), threshold};
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::global_cuda_stream_pool();
};

std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> output_mr_;

rmm::device_async_resource_ref get_output_mr() {
  return output_mr_.value();
}

AllocLabelGuard::AllocLabelGuard(char const* label)
    : previous_(allocLabel()) {
  allocLabel() = label;
}

AllocLabelGuard::~AllocLabelGuard() {
  allocLabel() = previous_;
}

} // namespace facebook::velox::cudf_velox

// This must NOT be in a file that includes CudfNoDefaults.h, because
// CudfNoDefaults.h redeclares cudf::get_default_stream() with
// __attribute__((error)). The overload below calls the real function.
namespace cudf {

rmm::cuda_stream_view const get_default_stream(allow_default_stream_t) {
  return cudf::get_default_stream();
}

} // namespace cudf
