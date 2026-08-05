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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {

/// Extract X (or Y) from Velox-serialized POINT geometry blobs stored as a
/// cuDF STRING/VARBINARY column. Non-POINT inputs set *invalidTypeFlag.
/// Empty POINTs (NaN coordinates) become null outputs.
///
/// Layout (GeometrySerde::writePoint): uint8 POINT tag + double x + double y.
std::unique_ptr<cudf::column> extractPointCoordinate(
    cudf::column_view const& geometry,
    bool extractY,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Euclidean point-point distance in coordinate units (matches CPU
/// ST_Distance for POINTs). Non-POINT inputs set *invalidTypeFlag. Empty
/// points yield null.
std::unique_ptr<cudf::column> pointPointDistance(
    cudf::column_view const& left,
    cudf::column_view const& right,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Build Velox POINT geometry blobs from x/y double columns.
std::unique_ptr<cudf::column> makePointGeometry(
    cudf::column_view const& x,
    cudf::column_view const& y,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Convert WKB POINT (2D, with optional EWKB SRID) to Velox POINT blobs.
/// Non-POINT / unsupported WKB sets *invalidTypeFlag and nulls the row.
std::unique_ptr<cudf::column> wkbPointToVeloxGeometry(
    cudf::column_view const& wkb,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
