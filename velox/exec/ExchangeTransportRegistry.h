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

#include "velox/core/PlanNode.h"
#include "velox/exec/ExchangeClient.h"
#include "velox/exec/Operator.h"

#include <functional>
#include <memory>
#include <string>

namespace facebook::velox::exec {

class DriverCtx;
class Task;

/// Type-erased exchange client owned by Task. HTTP wraps ExchangeClient; UCX
/// wraps UcxExchangeClient. Operators must not close the shared client.
class ExchangeClientHandle {
 public:
  virtual ~ExchangeClientHandle() = default;

  virtual void close() = 0;

  virtual void addRemoteTaskId(const std::string& remoteTaskId) = 0;

  virtual void noMoreRemoteTasks() = 0;

  virtual folly::dynamic toJson() const = 0;

  virtual std::shared_ptr<ExchangeClient> asHttpClient() const {
    return nullptr;
  }
};

std::shared_ptr<ExchangeClientHandle> wrapHttpExchangeClient(
    std::shared_ptr<ExchangeClient> client);

struct ExchangeTransportEntry {
  /// Creates the Task-owned client shared by every exchange operator in a
  /// pipeline. 'task' is non-null for HTTP so the client can use query pools.
  std::function<std::shared_ptr<ExchangeClientHandle>(
      const std::string& taskId,
      int destination,
      int32_t numberOfConsumers,
      Task* task,
      const core::PlanNodeId& planNodeId,
      int32_t pipelineId)>
      createClient;

  std::function<std::unique_ptr<Operator>(
      int32_t operatorId,
      DriverCtx* ctx,
      const std::shared_ptr<const core::ExchangeNode>& planNode,
      std::shared_ptr<ExchangeClientHandle> client)>
      createOperator;
};

class ExchangeTransportRegistry {
 public:
  static void registerTransport(std::string kind, ExchangeTransportEntry entry);

  static void unregisterTransport(std::string_view kind);

  static const ExchangeTransportEntry& get(std::string_view kind);

  static void registerHttpDefaults();
};

} // namespace facebook::velox::exec
