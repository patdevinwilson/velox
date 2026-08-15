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
#include <limits>
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

/// True if a sample of non-null rows looks like WKB/EWKB rather than Velox
/// geometry serde (tag 0/2/3/4/5/7). Used when join filters expose FieldAccess
/// to raw binary columns without an ST_GeomFromBinary wrapper.
bool geometryColumnLooksLikeWkb(
    cudf::column_view const& geometry,
    rmm::cuda_stream_view stream);

/// True if sampled non-null rows look like Velox POINT (tag 0, length 17) or
/// WKB Point. Used to resolve Within point-side when FieldAccess names collide.
bool geometryColumnLooksLikePoints(
    cudf::column_view const& geometry,
    rmm::cuda_stream_view stream);

/// Return Velox geometry: convert when `forceWkb` or the column looks like WKB.
/// nullptr means `geometry` is already Velox and can be used as-is.
std::unique_ptr<cudf::column> ensureVeloxGeometryColumn(
    cudf::column_view const& geometry,
    bool forceWkb,
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

/// ST_Within for POINT column vs constant POLYGON/ENVELOPE (even-odd rings,
/// shell and not in holes). Empty / non-POINT inputs → null / invalid flag.
std::unique_ptr<cudf::column> pointToConstantPolygonWithin(
    cudf::column_view const& points,
    DevicePolygonView polygon,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// ST_Within for a POINT column against a single POLYGON/ENVELOPE/MULTI_POLYGON
/// row (size-1 geometry column). Avoids broadcasting the polygon per candidate
/// (critical for SpatialBench Q10 zone ⟕ trips).
std::unique_ptr<cudf::column> geometryWithinPointsVsPolygon(
    cudf::column_view const& points,
    cudf::column_view const& polygon,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// ST_Within for candidate pairs without gathering geometry rows.
/// `pointIndices[i]` / `polyIndices[i]` select rows from `points` / `polygons`.
/// Avoids duplicating large multipolygon blobs (Q11 trip ⨝ zone).
std::unique_ptr<cudf::column> geometryWithinIndexed(
    cudf::column_view const& points,
    cudf::column_view const& polygons,
    cudf::column_view const& pointIndices,
    cudf::column_view const& polyIndices,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Like geometryWithinIndexed, but point coordinates are pre-extracted FLOAT64
/// columns (typically envelope minX/minY for point geometries). Skips per-
/// candidate Velox POINT blob parses — the dominant cost when millions of
/// candidates repeatedly touch the same 600M-point build (Q10/Q11).
std::unique_ptr<cudf::column> geometryWithinIndexedXY(
    cudf::column_view const& pointX,
    cudf::column_view const& pointY,
    cudf::column_view const& polygons,
    cudf::column_view const& pointIndices,
    cudf::column_view const& polyIndices,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// ST_Within for two geometry columns. Supports POINT in POLYGON/ENVELOPE
/// / MULTI_POLYGON in either argument order (SpatialJoin may flip sides).
/// Matches SpatialBench / Presto ST_Within(point, poly).
std::unique_ptr<cudf::column> geometryWithin(
    cudf::column_view const& left,
    cudf::column_view const& right,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// ST_Intersects for two geometry columns.
/// POINT–POLYGON/ENVELOPE (either order) → point-in-polygon.
/// POLYGON/ENVELOPE–POLYGON/ENVELOPE → envelope reject + vertex-in-poly +
/// shell edge crossings (SpatialBench Q6 bbox∩zone).
std::unique_ptr<cudf::column> geometryIntersects(
    cudf::column_view const& left,
    cudf::column_view const& right,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// ST_Intersects for a geometry column vs a constant POLYGON/ENVELOPE
/// (DevicePolygonView). Avoids broadcasting via make_column_from_scalar.
std::unique_ptr<cudf::column> geometryIntersectsConstantPolygon(
    cudf::column_view const& geometry,
    DevicePolygonView polygon,
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

/// ST_Area for Velox POLYGON / MULTI_POLYGON / ENVELOPE blobs (degree²,
/// matches CPU GEOS getArea). POINT / LINESTRING → 0. Signed shoelace over
/// all rings then abs — Esri CW exterior + CCW holes yields shell − holes.
std::unique_ptr<cudf::column> geometryArea(
    cudf::column_view const& geometry,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// ST_Intersection for POLYGON/ENVELOPE/MULTI_POLYGON pairs (exterior rings).
/// Uses Sutherland–Hodgman (convex clip); SpatialBench Q9 buildings are
/// simple convex footprints. Non-overlapping → empty polygon; unsupported
/// types set *invalidTypeFlag.
std::unique_ptr<cudf::column> geometryIntersection(
    cudf::column_view const& left,
    cudf::column_view const& right,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// geometry_union(array(geometry)) for SpatialBench Q5 point lists.
/// POINT-only lists → POINT (n=1) or MULTI_POINT (n≥2). All-empty → empty
/// POLYGON (GEOS/Presto parity). Mixed / non-POINT types set *invalidTypeFlag.
std::unique_ptr<cudf::column> geometryUnionFromList(
    cudf::column_view const& geometryLists,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// ST_ConvexHull for POINT / MULTI_POINT (SpatialBench Q5). Degenerate hulls
/// → POINT or LINESTRING; otherwise CW Esri POLYGON. Unsupported types or
/// >kMaxHullPoints vertices set *invalidTypeFlag for CPU fallback.
std::unique_ptr<cudf::column> geometryConvexHull(
    cudf::column_view const& geometry,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Per-row axis-aligned envelopes from Velox geometry STRING blobs.
/// POINT → (x,x,y,y); POLYGON/LINESTRING/MULTI_* → embedded envelope @ byte 5;
/// ENVELOPE → 4 doubles @ byte 1. Empty/NaN → null row.
/// If expandBy has size == geometry.size(), expands each envelope by that
/// row's radius (CPU SpatialJoinBuild::readEnvelope semantics). If expandBy
/// is empty, uses constantExpandBy.
struct GeometryEnvelopes {
  std::unique_ptr<cudf::column> minX;
  std::unique_ptr<cudf::column> minY;
  std::unique_ptr<cudf::column> maxX;
  std::unique_ptr<cudf::column> maxY;
};

GeometryEnvelopes extractGeometryEnvelopes(
    cudf::column_view const& geometry,
    cudf::column_view const& expandBy,
    double constantExpandBy,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Per-part (per-ring) envelope decomposition of a geometry column. A single
/// multipolygon whose parts straddle the antimeridian or scatter across islands
/// has a pathologically large whole-geometry bounding box (e.g. Alaska or "US
/// Minor Outlying Islands" span nearly the whole globe in X). Indexing each
/// ring's tight bbox instead lets the spatial grid reject the ~20% of trips
/// that fall in the coarse bbox but nowhere near the actual polygon.
///
/// Only outsized geometries are decomposed -- those whose bbox is far wider
/// than the column average. Splitting every polygon would inflate the index to
/// the column's total ring count, and query-side scratch is sized by the index
/// rather than by the batch, so an unfiltered split exhausts memory no matter
/// how small the probe batches get.
///
/// Returns per-part envelopes, `partToRow` mapping each part back to its source
/// row, and `rowPartOffset` (length numRows+1) giving each row's contiguous
/// part range. Points/envelopes/linestrings, ordinary-sized polygons, and
/// multipolygons with more than `maxPartsPerRow` rings collapse to a single
/// whole-geometry bbox part, so the index never explodes on pathological ring
/// counts.
struct GeometryPartEnvelopes {
  GeometryEnvelopes envelopes; // per part
  std::unique_ptr<cudf::column> partToRow; // size_type, length totalParts
  std::unique_ptr<cudf::column> rowPartOffset; // int64, length numRows + 1
};

GeometryPartEnvelopes extractGeometryPartEnvelopes(
    cudf::column_view const& geometry,
    cudf::column_view const& expandBy,
    double constantExpandBy,
    int32_t maxPartsPerRow,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Sort and drop duplicate (probeIdx, buildIdx) pairs. Used after per-part
/// candidate indices are remapped to row indices, where several rings of one
/// multipolygon can yield the same (row, row) pair.
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
dedupIndexPairs(
    std::unique_ptr<cudf::column> probeIdx,
    std::unique_ptr<cudf::column> buildIdx,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Cross-product envelope intersection: returns compacted (probeIndex,
/// buildIndex) pairs where probe and (already-expanded) build envelopes
/// intersect. Mirrors CPU SpatialIndex pruning before ST_Distance.
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
geometryEnvelopeCrossIntersectIndices(
    GeometryEnvelopes const& probe,
    GeometryEnvelopes const& build,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Uniform grid over build envelopes for O(candidates) probe queries.
/// Takes ownership of `buildEnvelopes`.
/// Builds whose envelopes span more than kMaxCellsPerBuild cells are kept in
/// `largeBuildIndices` and AABB-tested against every probe (avoids multi-GB
/// cell lists from continent-scale polygons).
struct GeometryEnvelopeGrid {
  static constexpr int32_t kMaxCellsPerBuild = 64;

  double originX{0};
  double originY{0};
  double invCellW{0};
  double invCellH{0};
  int32_t nCols{0};
  int32_t nRows{0};
  // True when every indexed envelope is a point (zero extent). Points land in
  // exactly one cell, so query results need no sort+unique dedupe — critical
  // for Q10's 600M-point build (cuSpatial quadtree leaves are similarly unique).
  bool isPointGrid{false};
  GeometryEnvelopes envelopes;
  std::unique_ptr<cudf::column> cellOffsets; // int64, nCells + 1
  std::unique_ptr<cudf::column> cellBuildIndices; // size_type
  std::unique_ptr<cudf::column> largeBuildIndices; // size_type, may be empty
};

GeometryEnvelopeGrid buildGeometryEnvelopeGrid(
    GeometryEnvelopes buildEnvelopes,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Probe the grid with probe envelopes; returns compacted (probeIdx, buildIdx)
/// pairs that pass AABB intersection (same contract as SpatialIndex::query).
///
/// The candidate count is known on the host after the counting pass but before
/// the pair arrays are allocated. Throws "match count overflow" when it exceeds
/// maxMatches, so callers can shrink the probe batch without ever attempting a
/// multi-GB allocation that would OOM the device.
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
queryGeometryEnvelopeGrid(
    GeometryEnvelopeGrid const& grid,
    GeometryEnvelopes const& probe,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    int64_t maxMatches = std::numeric_limits<int64_t>::max());

/// Exact Euclidean k-NN join indices for SpatialBench Q12 / ST_KNN.
/// Probe rows are typically POINTs; build rows are POLYGON/ENVELOPE (buildings).
/// Uses expanding envelope-grid search then exact ST_Distance ranking.
/// Returns compacted (probeIdx, buildIdx) with up to k neighbors per probe
/// (fewer if the build side has < k geometries). knnK must be in [1, 32].
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
geometryKnnJoinIndices(
    cudf::column_view const& probeGeometry,
    cudf::column_view const& buildGeometry,
    GeometryEnvelopeGrid const& buildGrid,
    int32_t knnK,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
