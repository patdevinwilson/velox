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

#include "velox/experimental/cudf/exec/CudfMarkDistinct.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <nvtx3/nvtx3.hpp>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

namespace facebook::velox::cudf_velox {

CudfMarkDistinct::CudfMarkDistinct(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::MarkDistinctNode> planNode)
    : exec::Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "CudfMarkDistinct"),
      NvtxHelper(
          nvtx3::rgb{128, 0, 128},
          operatorId,
          fmt::format("[{}]", planNode->id())),
      planNode_(planNode) {
  auto inputType = planNode->sources()[0]->outputType();
  for (const auto& key : planNode->distinctKeys()) {
    auto idx = inputType->getChildIdx(key->name());
    keyColumnIndices_.push_back(static_cast<cudf::size_type>(idx));
  }
}

void CudfMarkDistinct::addInput(RowVectorPtr input) {
  VELOX_CHECK_NULL(input_);
  input_ = std::move(input);
}

RowVectorPtr CudfMarkDistinct::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (input_ == nullptr) {
    return nullptr;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfMarkDistinct expects CudfVector");

  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto inputView = cudfInput->getTableView();
  const auto numRows = inputView.num_rows();

  input_.reset();

  if (numRows == 0) {
    return nullptr;
  }

  // Extract key columns from input
  auto keyView = inputView.select(keyColumnIndices_);

  // Strategy: concatenate seenKeys_ (if any) with this batch's keys, then
  // run distinct_indices on the combined table. Indices that fall in the
  // new batch range [seenCount, seenCount+numRows) are "first occurrence".
  const cudf::size_type seenCount =
      seenKeys_ ? seenKeys_->num_rows() : 0;

  std::unique_ptr<cudf::table> combinedKeys;
  if (seenKeys_ && seenCount > 0) {
    auto views = std::vector<cudf::table_view>{seenKeys_->view(), keyView};
    combinedKeys = cudf::concatenate(views, stream, mr);
  } else {
    combinedKeys = std::make_unique<cudf::table>(keyView, stream, mr);
  }

  // All key columns participate in distinct
  std::vector<cudf::size_type> allCols(combinedKeys->num_columns());
  std::iota(allCols.begin(), allCols.end(), 0);

  auto distinctIdx = cudf::distinct_indices(
      combinedKeys->view(),
      cudf::duplicate_keep_option::KEEP_FIRST,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream,
      mr);

  // Build boolean mask: for each row in [0, numRows), check if
  // (seenCount + row) appears in distinctIdx.
  // We do this on GPU: create a sorted copy of distinctIdx, then for each
  // row index in [seenCount, seenCount+numRows), binary search.
  auto falseScalar = cudf::make_numeric_scalar(
      cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
  falseScalar->set_valid_async(true, stream);
  static_cast<cudf::numeric_scalar<bool>&>(*falseScalar)
      .set_value(false, stream);
  auto maskCol = cudf::make_column_from_scalar(*falseScalar, numRows, stream, mr);

  // Filter distinctIdx to only indices in [seenCount, seenCount+numRows)
  // and mark those positions as true in the mask.
  auto trueScalar = cudf::make_numeric_scalar(
      cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
  trueScalar->set_valid_async(true, stream);
  static_cast<cudf::numeric_scalar<bool>&>(*trueScalar)
      .set_value(true, stream);

  // Convert distinctIdx device_uvector to a column for cuDF operations.
  rmm::device_buffer noNulls(0, stream, mr);
  auto distinctIdxCol = std::make_unique<cudf::column>(
      std::move(*distinctIdx), std::move(noNulls), 0);

  // Filter: keep only indices >= seenCount
  auto seenCountScalar = cudf::make_numeric_scalar(
      cudf::data_type{cudf::type_id::UINT32}, stream, mr);
  seenCountScalar->set_valid_async(true, stream);
  static_cast<cudf::numeric_scalar<uint32_t>&>(*seenCountScalar)
      .set_value(static_cast<uint32_t>(seenCount), stream);

  auto upperBoundScalar = cudf::make_numeric_scalar(
      cudf::data_type{cudf::type_id::UINT32}, stream, mr);
  upperBoundScalar->set_valid_async(true, stream);
  static_cast<cudf::numeric_scalar<uint32_t>&>(*upperBoundScalar)
      .set_value(static_cast<uint32_t>(seenCount + numRows), stream);

  // ge_mask: index >= seenCount
  auto geMask = cudf::binary_operation(
      distinctIdxCol->view(), *seenCountScalar,
      cudf::binary_operator::GREATER_EQUAL,
      cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
  // lt_mask: index < seenCount + numRows
  auto ltMask = cudf::binary_operation(
      distinctIdxCol->view(), *upperBoundScalar,
      cudf::binary_operator::LESS,
      cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
  // combined filter
  auto filterMask = cudf::binary_operation(
      geMask->view(), ltMask->view(),
      cudf::binary_operator::BITWISE_AND,
      cudf::data_type{cudf::type_id::BOOL8}, stream, mr);

  // Apply filter to get indices in the new batch range
  auto filteredTable = cudf::apply_boolean_mask(
      cudf::table_view({distinctIdxCol->view()}),
      filterMask->view(), stream, mr);
  auto filteredIndices = std::move(filteredTable->release()[0]);

  if (filteredIndices->size() > 0) {
    // Subtract seenCount to get local indices
    auto localIndices = cudf::binary_operation(
        filteredIndices->view(), *seenCountScalar,
        cudf::binary_operator::SUB,
        cudf::data_type{cudf::type_id::INT32}, stream, mr);

    // Scatter true into mask at local indices
    auto scatterTrue = cudf::make_column_from_scalar(
        *trueScalar, localIndices->size(), stream, mr);
    auto scatterTable = cudf::table_view({scatterTrue->view()});
    auto targetTable = cudf::table_view({maskCol->view()});
    auto scattered = cudf::scatter(
        scatterTable, localIndices->view(), targetTable, stream, mr);
    maskCol = std::move(scattered->release()[0]);
  }

  // Update seenKeys_: take distinct from the combined table
  seenKeys_ = cudf::distinct(
      combinedKeys->view(), allCols,
      cudf::duplicate_keep_option::KEEP_FIRST,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream, mr);

  // Build output: all input columns + mask column
  std::vector<std::unique_ptr<cudf::column>> outCols;
  outCols.reserve(inputView.num_columns() + 1);
  for (cudf::size_type i = 0; i < inputView.num_columns(); ++i) {
    outCols.push_back(
        std::make_unique<cudf::column>(inputView.column(i), stream, mr));
  }
  outCols.push_back(std::move(maskCol));

  auto outTable = std::make_unique<cudf::table>(std::move(outCols));
  stream.synchronize();

  return std::make_shared<CudfVector>(
      pool(), outputType_, outTable->num_rows(), std::move(outTable), stream);
}

} // namespace facebook::velox::cudf_velox
