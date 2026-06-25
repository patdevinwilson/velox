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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/type/Type.h"

#include <cudf/groupby.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace facebook::velox::cudf_velox {

/// GPU-accelerated Window operator using cuDF.
///
/// Default path: batches are buffered until noMoreInput(), concatenated and
/// sorted, then all window functions run in one getOutput() call.
///
/// Partition streaming (Phase 1 of #17917): when inputsSorted() is true and
/// partition keys are present, completed partitions are detected in addInput(),
/// queued, and processed one partition per getOutput() call. Peak memory scales
/// with the largest partition rather than total input size.
///
/// inputsSorted fast path: when WindowNode::inputsSorted() is true, this
/// operator skips stable_sorted_order (see WindowNode::inputsSorted() for the
/// ordering contract). The flag is taken from the plan as-is.
///
/// Rank-like functions (row_number, rank, dense_rank) use
/// cudf::groupby::scan with cudf::make_rank_aggregation.
/// Aggregate windows and lag/lead use cudf::grouped_rolling_window.
class CudfWindow : public CudfOperatorBase {
 public:
  CudfWindow(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowNode>& windowNode);

  /// Returns true if every window function and frame in the plan node is
  /// supported by CudfWindow. On failure, @p reason is set when provided.
  static bool canRunOnGPU(const core::WindowNode& windowNode);

  static bool canRunOnGPU(
      const core::WindowNode& windowNode,
      std::optional<std::string>& reason);

  /// Returns true if the window function is supported by CudfWindow.
  static bool isSupportedWindowFunction(
      const std::string& baseName,
      size_t numArgs) {
    static const std::unordered_set<std::string> kSupportedFuncs = {
        "lag",
        "lead",
        "row_number",
        "rank",
        "dense_rank",
        "first_value",
        "last_value",
        "sum",
        "min",
        "max",
        "count",
        "avg"};
    if (kSupportedFuncs.find(baseName) == kSupportedFuncs.end()) {
      return false;
    }
    // lag/lead only support up to 2 arguments (value, offset)
    if ((baseName == "lag" || baseName == "lead") && numArgs > 2) {
      return false;
    }
    return true;
  }

  bool needsInput() const override {
    if (noMoreInput_) {
      return false;
    }
    if (usePartitionStreaming_ && !completedPartitions_.empty()) {
      return false;
    }
    return true;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;

  RowVectorPtr doGetOutput() override;

  void doNoMoreInput() override;

  void doClose() override;

 private:
  // Compute row_number/rank/dense_rank via cudf::groupby::scan or cudf::scan.
  void computeRankColumnsBatch(
      const cudf::table_view& sortedInput,
      const std::vector<std::pair<size_t, std::string>>& pendingRanks,
      cudf::groupby::groupby* rankGrouper,
      std::vector<std::unique_ptr<cudf::column>>& windowResultCols,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Compute LAG or LEAD via cudf::grouped_rolling_window.
  std::unique_ptr<cudf::column> computeLeadLagColumn(
      const cudf::table_view& partKeys,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      const std::string& baseName,
      rmm::cuda_stream_view stream) const;

  // Compute first_value or last_value via cudf rolling window APIs.
  std::unique_ptr<cudf::column> computeNthValueColumn(
      const cudf::table_view& partKeys,
      const cudf::table_view& sortedView,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      const std::string& baseName,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Compute aggregate window functions (sum, min, max, count, avg)
  // with frame bounds from the WindowNode.
  std::unique_ptr<cudf::column> computeAggregateColumn(
      const cudf::table_view& partKeys,
      const cudf::table_view& sortedView,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      const std::string& baseName,
      bool isCountStar,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Dispatch to grouped_rolling_window (ROWS) or grouped_range_rolling_window
  // (RANGE) based on the frame type.
  std::unique_ptr<cudf::column> invokeGroupedRollingWindow(
      const cudf::table_view& partKeys,
      const cudf::table_view& sortedView,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      std::unique_ptr<cudf::rolling_aggregation> agg,
      bool isFullPartition,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  RowVectorPtr processWindowOnTable(
      std::unique_ptr<cudf::table> sortedData,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  void addStreamingInput(CudfVectorPtr batch);

  void finalizeCurrentPartition();

  std::shared_ptr<const core::WindowNode> windowNode_;
  const bool usePartitionStreaming_;
  const RowTypePtr inputRowType_;

  std::vector<cudf::size_type> partitionKeyIndices_;
  std::vector<cudf::size_type> sortKeyIndices_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> nullOrders_;

  std::vector<CudfVectorPtr> inputBatches_;

  // Partition streaming state (Phase 1, #17917).
  std::vector<CudfVectorPtr> currentPartitionBatches_;
  std::deque<std::unique_ptr<cudf::table>> completedPartitions_;

  // Bulk path: sorted and concatenated input, prepared in doNoMoreInput().
  std::unique_ptr<cudf::table> sortedData_;
  rmm::cuda_stream_view stream_{};

  bool finished_ = false;
};

} // namespace facebook::velox::cudf_velox
