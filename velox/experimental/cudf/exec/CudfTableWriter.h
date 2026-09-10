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

#include "velox/exec/TableWriter.h"

#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox {

/// Keeps CudfVector input resident on the GPU while the Hive Parquet sink
/// writes it. This is especially important for persistent CTE materialization:
/// the canonical TPC-DS Q47/Q57 plans otherwise convert the CTE producer to
/// CPU before TableWrite, forcing its entire upstream pipeline to fall back.
class CudfTableWriter final : public exec::TableWriter {
 public:
  CudfTableWriter(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const core::TableWriteNodePtr& tableWriteNode);

 protected:
  bool tryAppendInput(const RowVectorPtr& input) override;

 private:
  std::vector<cudf::size_type> inputMapping_;
};

} // namespace facebook::velox::cudf_velox
