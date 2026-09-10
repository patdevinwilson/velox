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

#include "velox/exec/ExchangeTransportRegistry.h"

#include <mutex>
#include <unordered_map>

#include "velox/core/PlanFragment.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {
namespace {

class HttpExchangeClientHandle final : public ExchangeClientHandle {
 public:
  explicit HttpExchangeClientHandle(std::shared_ptr<ExchangeClient> client)
      : client_(std::move(client)) {
    VELOX_CHECK_NOT_NULL(client_);
  }

  void close() override {
    client_->close();
  }

  void addRemoteTaskId(const std::string& remoteTaskId) override {
    client_->addRemoteTaskId(remoteTaskId);
  }

  void noMoreRemoteTasks() override {
    client_->noMoreRemoteTasks();
  }

  folly::dynamic toJson() const override {
    return client_->toJson();
  }

  std::shared_ptr<ExchangeClient> asHttpClient() const override {
    return client_;
  }

 private:
  const std::shared_ptr<ExchangeClient> client_;
};

std::mutex& mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, ExchangeTransportEntry>& entries() {
  static std::unordered_map<std::string, ExchangeTransportEntry> map;
  return map;
}

ExchangeTransportEntry makeHttpEntry() {
  ExchangeTransportEntry entry;
  entry.createClient = nullptr;
  entry.createOperator =
      [](int32_t operatorId,
         DriverCtx* ctx,
         const std::shared_ptr<const core::ExchangeNode>& planNode,
         std::shared_ptr<ExchangeClientHandle> client) {
        VELOX_CHECK_NOT_NULL(client);
        auto http = client->asHttpClient();
        VELOX_CHECK_NOT_NULL(http, "HTTP exchange requires an HTTP client");
        return std::make_unique<Exchange>(
            operatorId, ctx, planNode, std::move(http));
      };
  return entry;
}

} // namespace

std::shared_ptr<ExchangeClientHandle> wrapHttpExchangeClient(
    std::shared_ptr<ExchangeClient> client) {
  return std::make_shared<HttpExchangeClientHandle>(std::move(client));
}

void ExchangeTransportRegistry::registerHttpDefaults() {
  std::lock_guard<std::mutex> lock(mutex());
  auto& map = entries();
  if (map.find(std::string{core::TransportKind::kHttp}) == map.end()) {
    map.emplace(std::string{core::TransportKind::kHttp}, makeHttpEntry());
  }
}

void ExchangeTransportRegistry::registerTransport(
    std::string kind,
    ExchangeTransportEntry entry) {
  registerHttpDefaults();
  std::lock_guard<std::mutex> lock(mutex());
  entries()[std::move(kind)] = std::move(entry);
}

void ExchangeTransportRegistry::unregisterTransport(std::string_view kind) {
  std::lock_guard<std::mutex> lock(mutex());
  entries().erase(std::string{kind});
}

const ExchangeTransportEntry& ExchangeTransportRegistry::get(
    std::string_view kind) {
  registerHttpDefaults();
  std::lock_guard<std::mutex> lock(mutex());
  auto it = entries().find(std::string{kind});
  VELOX_USER_CHECK(
      it != entries().end(),
      "No exchange transport registered for kind {}",
      kind);
  return it->second;
}

} // namespace facebook::velox::exec
