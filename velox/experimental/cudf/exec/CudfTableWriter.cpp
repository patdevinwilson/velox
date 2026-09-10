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

#include "velox/experimental/cudf/exec/CudfTableWriter.h"

#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSink.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::cudf_velox {

CudfTableWriter::CudfTableWriter(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const core::TableWriteNodePtr& tableWriteNode)
    : exec::TableWriter(
          operatorId,
          driverCtx,
          tableWriteNode,
          "CudfTableWrite") {
  const auto& inputType = tableWriteNode->sources()[0]->outputType();
  inputMapping_.reserve(tableWriteNode->columns()->size());
  for (const auto& name : tableWriteNode->columns()->names()) {
    inputMapping_.push_back(inputType->getChildIdx(name));
  }
}

bool CudfTableWriter::tryAppendInput(const RowVectorPtr& input) {
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  if (!cudfInput) {
    return false;
  }

  auto* sink = dynamic_cast<connector::hive::CudfHiveDataSink*>(dataSink());
  VELOX_CHECK_NOT_NULL(
      sink, "CudfTableWriter requires a CudfHiveDataSink");
  sink->appendCudfData(
      cudfInput->getTableView().select(inputMapping_),
      cudfInput->stream(),
      input->estimateFlatSize());
  return true;
}

} // namespace facebook::velox::cudf_velox
