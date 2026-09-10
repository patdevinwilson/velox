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

#include "velox/exec/OutputTransportRegistry.h"

#include <mutex>
#include <unordered_map>

#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {
namespace {

std::mutex& mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, OutputTransportEntry>& entries() {
  static std::unordered_map<std::string, OutputTransportEntry> map;
  return map;
}

OutputTransportEntry makeHttpEntry() {
  OutputTransportEntry entry;
  entry.initializeTask =
      [](std::shared_ptr<Task> task,
         core::PartitionedOutputNode::Kind kind,
         int numDestinations,
         int numDrivers) {
        DefaultOutputBufferManager::getInstanceRef()->initializeTask(
            std::move(task), kind, numDestinations, numDrivers);
      };
  entry.removeTask = [](const std::string& taskId) {
    DefaultOutputBufferManager::getInstanceRef()->removeTask(taskId);
  };
  entry.updateOutputBuffers =
      [](const std::string& taskId, int numBuffers, bool noMore) {
        return DefaultOutputBufferManager::getInstanceRef()->updateOutputBuffers(
            taskId, numBuffers, noMore);
      };
  entry.updateNumDrivers =
      [](const std::string& taskId, uint32_t newNumDrivers) {
        return DefaultOutputBufferManager::getInstanceRef()->updateNumDrivers(
            taskId, newNumDrivers);
      };
  entry.stats = [](const std::string& taskId) {
    return DefaultOutputBufferManager::getInstanceRef()->stats(taskId);
  };
  entry.getUtilization = [](const std::string& taskId) {
    return DefaultOutputBufferManager::getInstanceRef()->getUtilization(
        taskId);
  };
  entry.isOverutilized = [](const std::string& taskId) {
    return DefaultOutputBufferManager::getInstanceRef()->isOverutilized(taskId);
  };
  entry.createOperator =
      [](int32_t operatorId,
         DriverCtx* ctx,
         const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
         bool eagerFlush) {
        return std::make_unique<PartitionedOutput>(
            operatorId, ctx, planNode, eagerFlush);
      };
  return entry;
}

} // namespace

void OutputTransportRegistry::registerHttpDefaults() {
  std::lock_guard<std::mutex> lock(mutex());
  auto& map = entries();
  if (map.find(std::string{core::TransportKind::kHttp}) == map.end()) {
    map.emplace(std::string{core::TransportKind::kHttp}, makeHttpEntry());
  }
}

void OutputTransportRegistry::registerTransport(
    std::string kind,
    OutputTransportEntry entry) {
  registerHttpDefaults();
  std::lock_guard<std::mutex> lock(mutex());
  entries()[std::move(kind)] = std::move(entry);
}

void OutputTransportRegistry::unregisterTransport(std::string_view kind) {
  std::lock_guard<std::mutex> lock(mutex());
  entries().erase(std::string{kind});
}

const OutputTransportEntry& OutputTransportRegistry::get(
    std::string_view kind) {
  registerHttpDefaults();
  std::lock_guard<std::mutex> lock(mutex());
  auto it = entries().find(std::string{kind});
  VELOX_USER_CHECK(
      it != entries().end(),
      "No output transport registered for kind {}",
      kind);
  return it->second;
}

} // namespace facebook::velox::exec
