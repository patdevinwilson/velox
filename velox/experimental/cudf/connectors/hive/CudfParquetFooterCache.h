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

#include <cudf/io/parquet_schema.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// Process-wide-per-connector cache of parsed Parquet FileMetaData.
///
/// Hive splits of a single large file each construct a CudfSplitReader, and
/// each used to call cudf::io::read_parquet_footers. For a 147GB file with
/// tens of thousands of row groups that parse is hundreds of milliseconds and
/// dominates scan wall time once files are splittable.
///
/// Entries are keyed by file path. Concurrent first loads of the same path
/// share one parse via std::once_flag.
class CudfParquetFooterCache {
 public:
  using Footers = std::vector<cudf::io::parquet::FileMetaData>;
  using Loader = std::function<Footers()>;

  explicit CudfParquetFooterCache(size_t maxFiles = 64);

  /// Returns a copy of the cached footers, loading them on a miss.
  Footers getOrLoad(const std::string& path, const Loader& loader);

  size_t hits() const {
    return hits_.load(std::memory_order_relaxed);
  }
  size_t misses() const {
    return misses_.load(std::memory_order_relaxed);
  }
  size_t size() const;

  void clear();

 private:
  struct Entry {
    std::once_flag once;
    Footers footers;
    std::exception_ptr error;
  };

  std::shared_ptr<Entry> getOrCreateEntry(const std::string& path);
  void maybeEvictLocked();

  const size_t maxFiles_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Entry>> entries_;
  std::vector<std::string> insertionOrder_;
  std::atomic<size_t> hits_{0};
  std::atomic<size_t> misses_{0};
};

} // namespace facebook::velox::cudf_velox::connector::hive
