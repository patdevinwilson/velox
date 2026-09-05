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

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

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

/// Convert WKB POINT/POLYGON (2D, optional EWKB SRID) to Velox geometry blobs.
/// Unsupported / malformed WKB sets *invalidTypeFlag and nulls the row.
std::unique_ptr<cudf::column> wkbToVeloxGeometry(
    cudf::column_view const& wkb,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Device view of a constant polygon (shell = part 0, holes = rest).
struct DevicePolygonView {
  double const* xy{nullptr}; // interleaved x,y ; length 2 * numPoints
  int32_t const* partEnds{nullptr}; // exclusive end point index per part
  int32_t numParts{0};
  int32_t numPoints{0};
};

/// Host-side parse of a Velox POLYGON or ENVELOPE blob into ring coordinates.
/// Returns false if the type is unsupported (non-polygon / empty).
bool parseVeloxPolygon(
    std::string_view geometry,
    std::vector<double>& xyOut,
    std::vector<int32_t>& partEndsOut);

/// Euclidean distance from Velox POINT column to a constant polygon.
/// Points inside the shell and outside all holes yield 0. Non-POINT inputs
/// set *invalidTypeFlag.
std::unique_ptr<cudf::column> pointToConstantPolygonDistance(
    cudf::column_view const& points,
    DevicePolygonView polygon,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Euclidean ST_Distance for two geometry columns. Supports POINT–POINT and
/// POINT–POLYGON (either side). Degrees; matches CPU SpatialBench semantics.
std::unique_ptr<cudf::column> geometryDistance(
    cudf::column_view const& left,
    cudf::column_view const& right,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Build Velox LINESTRING blobs from a LIST of POINT geometry blobs
/// (ST_LineString). Lists with < 2 points → empty LINESTRING. Non-POINT /
/// empty / null / repeated consecutive points set *invalidTypeFlag.
std::unique_ptr<cudf::column> makeLineStringFromPointList(
    cudf::column_view const& pointLists,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Euclidean ST_Length for Velox LINESTRING / MULTI_LINE_STRING blobs
/// (degree-space, matches CPU SpatialBench / GEOS getLength).
std::unique_ptr<cudf::column> lineStringLength(
    cudf::column_view const& geometry,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Great-circle distance between (lat1, lon1) and (lat2, lon2), computed via
/// cuSpatial's header-only `cuspatial::haversine_distance` kernel (see
/// GeometryKernels.cu for why cuSpatial is vendored header-only). Matches
/// BingTileType::greatCircleDistance's radius constant by default (pass the
/// same `radiusKm` used there). Result is null wherever any input is null.
std::unique_ptr<cudf::column> haversineGreatCircleDistance(
    cudf::column_view const& lat1,
    cudf::column_view const& lon1,
    cudf::column_view const& lat2,
    cudf::column_view const& lon2,
    double radiusKm,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
