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

#pragma once

#include <cudf/detail/utilities/stream_pool.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <optional>
#include <string_view>

namespace facebook::velox::cudf_velox {

extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
    output_mr_;

/// Returns the memory resource designated for output vector allocations.
rmm::device_async_resource_ref get_output_mr();

/// Names the operation allocating on this thread so the big-allocation tracer
/// can attribute an outsized request to a call site. Symbolized stack traces
/// are ambiguous under inlining; a label is not.
class AllocLabelGuard {
 public:
  explicit AllocLabelGuard(char const* label);
  ~AllocLabelGuard();

  AllocLabelGuard(AllocLabelGuard const&) = delete;
  AllocLabelGuard& operator=(AllocLabelGuard const&) = delete;

 private:
  char const* previous_;
};

/**
 * @brief Creates a memory resource based on the given mode.
 *
 * @param mode rmm::mr::pool_memory_resource mode.
 * @param percent The initial percent of GPU memory to allocate for memory
 * resource.
 * @param maxPercent Upper bound on managed pool growth, as a percent of total
 * device memory; values above 100 oversubscribe into host RAM. Ignored by
 * modes whose pool cannot outgrow the device. 0 means unbounded.
 */
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
createMemoryResource(std::string_view mode, int percent, int maxPercent = 0);

/**
 * @brief Returns the global CUDA stream pool used by cudf.
 */
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

} // namespace facebook::velox::cudf_velox
