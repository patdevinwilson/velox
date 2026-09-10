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

#include "velox/core/PlanFragment.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OutputBuffer.h"

#include <functional>
#include <optional>
#include <string>

namespace facebook::velox::exec {

class DriverCtx;
class Task;

/// Looks up a fragment-level input transport annotation. Missing keys default
/// to HTTP so unannotated plans keep the existing shuffle path.
inline std::string inputTransportKind(
    const core::PlanFragment& fragment,
    const core::PlanNodeId& planNodeId) {
  auto it = fragment.inputTransportTypes.find(planNodeId);
  return it == fragment.inputTransportTypes.end()
      ? std::string{core::TransportKind::kHttp}
      : it->second;
}

inline std::string outputTransportKind(
    const core::PlanFragment& fragment,
    const core::PlanNodeId& planNodeId) {
  auto it = fragment.outputTransportTypes.find(planNodeId);
  return it == fragment.outputTransportTypes.end()
      ? std::string{core::TransportKind::kHttp}
      : it->second;
}

/// Process-wide mapping from transport kind to partitioned-output operators
/// and the matching output-buffer manager. HTTP is registered by default.
/// Experimental transports (UCX) register from their own library.
struct OutputTransportEntry {
  std::function<void(
      std::shared_ptr<Task>,
      core::PartitionedOutputNode::Kind,
      int numDestinations,
      int numDrivers)>
      initializeTask;
  std::function<void(const std::string& taskId)> removeTask;
  std::function<bool(const std::string& taskId, int numBuffers, bool noMore)>
      updateOutputBuffers;
  std::function<bool(const std::string& taskId, uint32_t newNumDrivers)>
      updateNumDrivers;
  std::function<std::optional<OutputBuffer::Stats>(const std::string& taskId)>
      stats;
  std::function<double(const std::string& taskId)> getUtilization;
  std::function<bool(const std::string& taskId)> isOverutilized;
  std::function<std::unique_ptr<Operator>(
      int32_t operatorId,
      DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
      bool eagerFlush)>
      createOperator;
};

class OutputTransportRegistry {
 public:
  static void registerTransport(std::string kind, OutputTransportEntry entry);

  static void unregisterTransport(std::string_view kind);

  /// Throws if 'kind' is unknown. Callers must not silently fall back to HTTP.
  static const OutputTransportEntry& get(std::string_view kind);

  static void registerHttpDefaults();
};

} // namespace facebook::velox::exec
