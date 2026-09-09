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

#include "velox/experimental/cudf/connectors/hive/CudfParquetFooterCache.h"

#include "velox/common/base/Exceptions.h"

#include <algorithm>

namespace facebook::velox::cudf_velox::connector::hive {

CudfParquetFooterCache::CudfParquetFooterCache(size_t maxFiles)
    : maxFiles_(maxFiles == 0 ? 64 : maxFiles) {}

std::shared_ptr<CudfParquetFooterCache::Entry>
CudfParquetFooterCache::getOrCreateEntry(const std::string& path) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(path);
  if (it != entries_.end()) {
    return it->second;
  }
  maybeEvictLocked();
  auto entry = std::make_shared<Entry>();
  entries_.emplace(path, entry);
  insertionOrder_.push_back(path);
  return entry;
}

void CudfParquetFooterCache::maybeEvictLocked() {
  while (entries_.size() >= maxFiles_ && !insertionOrder_.empty()) {
    const auto victim = insertionOrder_.front();
    insertionOrder_.erase(insertionOrder_.begin());
    entries_.erase(victim);
  }
}

CudfParquetFooterCache::Footers CudfParquetFooterCache::getOrLoad(
    const std::string& path,
    const Loader& loader) {
  VELOX_CHECK(!path.empty(), "Parquet footer cache key must not be empty");
  VELOX_CHECK(loader, "Parquet footer cache loader must not be empty");

  auto entry = getOrCreateEntry(path);
  bool loadedHere = false;
  std::call_once(entry->once, [&]() {
    loadedHere = true;
    try {
      entry->footers = loader();
    } catch (...) {
      entry->error = std::current_exception();
    }
  });
  if (loadedHere) {
    misses_.fetch_add(1, std::memory_order_relaxed);
  } else {
    hits_.fetch_add(1, std::memory_order_relaxed);
  }
  if (entry->error) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = entries_.find(path);
      if (it != entries_.end() && it->second == entry) {
        entries_.erase(it);
        insertionOrder_.erase(
            std::remove(
                insertionOrder_.begin(), insertionOrder_.end(), path),
            insertionOrder_.end());
      }
    }
    std::rethrow_exception(entry->error);
  }
  return entry->footers;
}

size_t CudfParquetFooterCache::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}

void CudfParquetFooterCache::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.clear();
  insertionOrder_.clear();
}

} // namespace facebook::velox::cudf_velox::connector::hive
