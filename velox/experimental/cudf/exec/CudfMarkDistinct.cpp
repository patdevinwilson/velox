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

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <nvtx3/nvtx3.hpp>

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

  // Create boolean mask: all true initially
  auto allTrue = cudf::make_numeric_scalar(
      cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
  allTrue->set_valid_async(true, stream);
  static_cast<cudf::numeric_scalar<bool>&>(*allTrue).set_value(true, stream);
  auto maskCol = cudf::make_column_from_scalar(*allTrue, numRows, stream, mr);

  if (seenKeys_ != nullptr && seenKeys_->num_rows() > 0) {
    // Find rows whose keys already exist in seenKeys_ via left_semi_join
    auto seenView = seenKeys_->view();
    std::vector<cudf::size_type> leftOn(keyColumnIndices_.size());
    std::vector<cudf::size_type> rightOn(keyColumnIndices_.size());
    std::iota(leftOn.begin(), leftOn.end(), 0);
    std::iota(rightOn.begin(), rightOn.end(), 0);

    auto matchIndices = cudf::left_semi_join(
        keyView, seenView, leftOn, rightOn,
        cudf::null_equality::EQUAL, stream, mr);

    // Set mask to false for rows that matched (already seen)
    if (matchIndices->size() > 0) {
      auto falseScalar = cudf::make_numeric_scalar(
          cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
      falseScalar->set_valid_async(true, stream);
      static_cast<cudf::numeric_scalar<bool>&>(*falseScalar)
          .set_value(false, stream);

      auto indicesCol = std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::INT32},
          matchIndices->size(),
          matchIndices->release(),
          rmm::device_buffer{},
          0);
      auto scatterSrc = cudf::make_column_from_scalar(
          *falseScalar, matchIndices->size(), stream, mr);
      auto scatterTable = cudf::table_view({scatterSrc->view()});
      auto targetTable = cudf::table_view({maskCol->view()});
      auto scattered =
          cudf::scatter(scatterTable, indicesCol->view(), targetTable,
                        stream, mr);
      maskCol = std::move(scattered->release()[0]);
    }
  }

  // Within this batch, handle intra-batch duplicates: for rows with keys
  // that appear earlier in the same batch and are NOT already in seenKeys_,
  // only the first occurrence should be true.
  // Use cudf::distinct to find first-occurrence indices within the batch keys.
  auto distinctIndices = cudf::distinct_indices(
      keyView,
      cudf::duplicate_keep_option::KEEP_FIRST,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream,
      mr);

  if (distinctIndices->size() < static_cast<std::size_t>(numRows)) {
    // Some intra-batch duplicates exist. Build a "is first occurrence" mask.
    auto firstOccurrence = cudf::make_numeric_scalar(
        cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
    firstOccurrence->set_valid_async(true, stream);
    static_cast<cudf::numeric_scalar<bool>&>(*firstOccurrence)
        .set_value(false, stream);
    auto intraMask = cudf::make_column_from_scalar(
        *firstOccurrence, numRows, stream, mr);

    auto trueScalar = cudf::make_numeric_scalar(
        cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
    trueScalar->set_valid_async(true, stream);
    static_cast<cudf::numeric_scalar<bool>&>(*trueScalar)
        .set_value(true, stream);

    auto distinctIndicesCol = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        distinctIndices->size(),
        distinctIndices->release(),
        rmm::device_buffer{},
        0);
    auto scatterTrue = cudf::make_column_from_scalar(
        *trueScalar, distinctIndicesCol->size(), stream, mr);
    auto scatterTable = cudf::table_view({scatterTrue->view()});
    auto targetTable = cudf::table_view({intraMask->view()});
    auto scattered =
        cudf::scatter(scatterTable, distinctIndicesCol->view(), targetTable,
                      stream, mr);
    auto intraMaskResult = std::move(scattered->release()[0]);

    // AND the intra-batch mask with the cross-batch mask
    maskCol = cudf::binary_operation(
        maskCol->view(), intraMaskResult->view(),
        cudf::binary_operator::BITWISE_AND,
        cudf::data_type{cudf::type_id::BOOL8},
        stream, mr);
  }

  // Update seenKeys_ with all distinct keys from this batch
  auto newKeysTable = std::make_unique<cudf::table>(keyView, stream, mr);
  auto newDistinct = cudf::distinct(
      newKeysTable->view(),
      std::vector<cudf::size_type>(
          keyColumnIndices_.size()),
      cudf::duplicate_keep_option::KEEP_FIRST,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream, mr);

  if (seenKeys_ != nullptr && seenKeys_->num_rows() > 0) {
    auto views = std::vector<cudf::table_view>{
        seenKeys_->view(), newDistinct->view()};
    auto combined = cudf::concatenate(views, stream, mr);
    // Deduplicate the combined table
    std::vector<cudf::size_type> allCols(combined->num_columns());
    std::iota(allCols.begin(), allCols.end(), 0);
    seenKeys_ = cudf::distinct(
        combined->view(), allCols,
        cudf::duplicate_keep_option::KEEP_FIRST,
        cudf::null_equality::EQUAL,
        cudf::nan_equality::ALL_EQUAL,
        stream, mr);
  } else {
    seenKeys_ = std::move(newDistinct);
  }

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
