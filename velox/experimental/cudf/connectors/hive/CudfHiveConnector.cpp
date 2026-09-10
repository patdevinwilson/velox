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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSink.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/HiveDataSink.h"

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;

CudfHiveConnector::CudfHiveConnector(
    const std::string& id,
    std::shared_ptr<const facebook::velox::config::ConfigBase> config,
    folly::Executor* executor)
    : ::facebook::velox::connector::hive::HiveConnector(id, config, executor),
      cudfHiveConfig_(std::make_shared<CudfHiveConfig>(config)) {
  VLOG(1) << "cuDF Hive connector created";
}

std::unique_ptr<DataSource> CudfHiveConnector::createDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) {
  // If it's parquet then return CudfHiveDataSource
  // If it's not parquet then return HiveDataSource
  // TODO (dm): Make this ^^^ happen
  // Problem: this information is in split, not table handle

  if (cudfIsRegistered()) {
    return std::make_unique<CudfHiveDataSource>(
        outputType,
        tableHandle,
        columnHandles,
        &fileHandleFactory_,
        ioExecutor_,
        connectorQueryCtx,
        cudfHiveConfig_);
  }

  return std::make_unique<::facebook::velox::connector::hive::HiveDataSource>(
      outputType,
      tableHandle,
      columnHandles,
      &fileHandleFactory_,
      ioExecutor_,
      connectorQueryCtx,
      hiveConfig_);
}

std::unique_ptr<DataSink> CudfHiveConnector::createDataSink(
    RowTypePtr inputType,
    ConnectorInsertTableHandlePtr connectorInsertTableHandle,
    ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy) {
  auto cudfHiveInsertHandle =
      std::dynamic_pointer_cast<const CudfHiveInsertTableHandle>(
          connectorInsertTableHandle);
  if (!cudfHiveInsertHandle) {
    auto hiveInsertHandle = std::dynamic_pointer_cast<
        const ::facebook::velox::connector::hive::HiveInsertTableHandle>(
        connectorInsertTableHandle);
    VELOX_CHECK_NOT_NULL(
        hiveInsertHandle,
        "cuDF Hive connector expects a Hive write handle");
    VELOX_USER_CHECK_EQ(
        hiveInsertHandle->storageFormat(),
        dwio::common::FileFormat::PARQUET,
        "cuDF Hive data sink only supports PARQUET");

    std::vector<std::shared_ptr<const CudfHiveColumnHandle>> inputColumns;
    inputColumns.reserve(hiveInsertHandle->inputColumns().size());
    for (const auto& column : hiveInsertHandle->inputColumns()) {
      inputColumns.push_back(std::make_shared<const CudfHiveColumnHandle>(
          column->name(),
          column->dataType(),
          veloxToCudfDataType(column->dataType())));
    }
    auto location = std::make_shared<const LocationHandle>(
        hiveInsertHandle->locationHandle()->targetPath(),
        LocationHandle::TableType::kNew,
        hiveInsertHandle->locationHandle()->targetFileName());
    cudfHiveInsertHandle = std::make_shared<const CudfHiveInsertTableHandle>(
        std::move(inputColumns),
        std::move(location),
        hiveInsertHandle->compressionKind(),
        hiveInsertHandle->serdeParameters(),
        hiveInsertHandle->writerOptions());
  }
  VELOX_CHECK_NOT_NULL(
      cudfHiveInsertHandle,
      "cuDF Hive connector expects a cuDF Hive write handle");
  return std::make_unique<CudfHiveDataSink>(
      std::move(inputType),
      std::move(cudfHiveInsertHandle),
      connectorQueryCtx,
      commitStrategy,
      cudfHiveConfig_);
}

std::shared_ptr<Connector> CudfHiveConnectorFactory::newConnector(
    const std::string& id,
    std::shared_ptr<const facebook::velox::config::ConfigBase> config,
    folly::Executor* ioExecutor,
    folly::Executor* cpuExecutor) {
  return std::make_shared<CudfHiveConnector>(id, config, ioExecutor);
}

} // namespace facebook::velox::cudf_velox::connector::hive
