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

#include "velox/experimental/cudf/expression/GeometryKernels.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

// Must match GeometrySerializationType::POINT in GeometryConstants.h.
constexpr uint8_t kPointTag = 0;
constexpr int32_t kPointBlobSize = 17; // 1 + 8 + 8

__device__ inline void markInvalid(int32_t* flag) {
  if (flag != nullptr) {
    atomicOr(flag, 1);
  }
}

__device__ inline bool readPointXY(
    char const* data,
    cudf::size_type len,
    double& x,
    double& y,
    int32_t* invalidTypeFlag) {
  if (len < kPointBlobSize ||
      static_cast<uint8_t>(data[0]) != kPointTag) {
    markInvalid(invalidTypeFlag);
    return false;
  }
  // Device memcpy of unaligned doubles from the geometry blob.
  char xb[8];
  char yb[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    xb[i] = data[1 + i];
    yb[i] = data[9 + i];
  }
  {
    double tmpX = 0;
    double tmpY = 0;
    auto* xp = reinterpret_cast<char*>(&tmpX);
    auto* yp = reinterpret_cast<char*>(&tmpY);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      xp[i] = xb[i];
      yp[i] = yb[i];
    }
    x = tmpX;
    y = tmpY;
  }
  return true;
}

__device__ inline bool isEmptyPoint(double x, double y) {
  return isnan(x) && isnan(y);
}

__device__ inline uint32_t readU32(char const* p, bool littleEndian) {
  uint8_t b0 = static_cast<uint8_t>(p[0]);
  uint8_t b1 = static_cast<uint8_t>(p[1]);
  uint8_t b2 = static_cast<uint8_t>(p[2]);
  uint8_t b3 = static_cast<uint8_t>(p[3]);
  if (littleEndian) {
    return uint32_t{b0} | (uint32_t{b1} << 8) | (uint32_t{b2} << 16) |
        (uint32_t{b3} << 24);
  }
  return uint32_t{b3} | (uint32_t{b2} << 8) | (uint32_t{b1} << 16) |
      (uint32_t{b0} << 24);
}

__device__ inline double readF64(char const* p, bool littleEndian) {
  char bytes[8];
  if (littleEndian) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      bytes[i] = p[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      bytes[i] = p[7 - i];
    }
  }
  double out = 0;
  auto* outb = reinterpret_cast<char*>(&out);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    outb[i] = bytes[i];
  }
  return out;
}

/// Parse 2D WKB/EWKB Point into Velox POINT blob bytes at out[17].
/// Returns false on unsupported / truncated input.
__device__ inline bool wkbPointToVeloxBlob(
    char const* wkb,
    cudf::size_type len,
    char* out,
    int32_t* invalidTypeFlag) {
  // Minimal 2D Point WKB: endian(1) + type(4) + x(8) + y(8) = 21.
  if (len < 21) {
    markInvalid(invalidTypeFlag);
    return false;
  }
  uint8_t const byteOrder = static_cast<uint8_t>(wkb[0]);
  if (byteOrder > 1) {
    markInvalid(invalidTypeFlag);
    return false;
  }
  bool const le = byteOrder == 1;
  uint32_t typeWord = readU32(wkb + 1, le);
  // Strip EWKB high flags; keep ISO type in low 8 bits for Point==1.
  constexpr uint32_t kSridFlag = 0x20000000u;
  bool const hasSrid = (typeWord & kSridFlag) != 0;
  uint32_t const geomType = typeWord & 0xFFu;
  if (geomType != 1) {
    // Only 2D Point (type 1). PointZ/M etc. rejected for Phase 1.
    markInvalid(invalidTypeFlag);
    return false;
  }
  cudf::size_type coordOffset = 5;
  if (hasSrid) {
    if (len < 25) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    coordOffset = 9;
  }
  double const x = readF64(wkb + coordOffset, le);
  double const y = readF64(wkb + coordOffset + 8, le);
  out[0] = static_cast<char>(kPointTag);
  auto const* xb = reinterpret_cast<char const*>(&x);
  auto const* yb = reinterpret_cast<char const*>(&y);
#pragma unroll
  for (int b = 0; b < 8; ++b) {
    out[1 + b] = xb[b];
    out[9 + b] = yb[b];
  }
  return true;
}

__device__ inline double distPointSegment(
    double px,
    double py,
    double ax,
    double ay,
    double bx,
    double by) {
  double const dx = bx - ax;
  double const dy = by - ay;
  double const len2 = dx * dx + dy * dy;
  if (len2 == 0.0) {
    double const ex = px - ax;
    double const ey = py - ay;
    return sqrt(ex * ex + ey * ey);
  }
  double t = ((px - ax) * dx + (py - ay) * dy) / len2;
  t = fmin(1.0, fmax(0.0, t));
  double const qx = ax + t * dx;
  double const qy = ay + t * dy;
  double const ex = px - qx;
  double const ey = py - qy;
  return sqrt(ex * ex + ey * ey);
}

/// Even-odd point-in-ring. partStart/partEnd are point indices [start, end).
__device__ inline bool pointInRing(
    double px,
    double py,
    double const* xy,
    int32_t partStart,
    int32_t partEnd) {
  if (partEnd - partStart < 3) {
    return false;
  }
  bool inside = false;
  for (int32_t i = partStart, j = partEnd - 1; i < partEnd; j = i++) {
    double const xi = xy[2 * i];
    double const yi = xy[2 * i + 1];
    double const xj = xy[2 * j];
    double const yj = xy[2 * j + 1];
    bool const intersect = ((yi > py) != (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi + 0.0) + xi);
    if (intersect) {
      inside = !inside;
    }
  }
  return inside;
}

__device__ inline double minDistToRing(
    double px,
    double py,
    double const* xy,
    int32_t partStart,
    int32_t partEnd) {
  double best = INFINITY;
  if (partEnd - partStart < 2) {
    return best;
  }
  for (int32_t i = partStart; i < partEnd - 1; ++i) {
    double const d = distPointSegment(
        px,
        py,
        xy[2 * i],
        xy[2 * i + 1],
        xy[2 * (i + 1)],
        xy[2 * (i + 1) + 1]);
    best = fmin(best, d);
  }
  // Close the ring if first != last (still safe if already closed).
  int32_t const last = partEnd - 1;
  double const dClose = distPointSegment(
      px,
      py,
      xy[2 * last],
      xy[2 * last + 1],
      xy[2 * partStart],
      xy[2 * partStart + 1]);
  return fmin(best, dClose);
}

} // namespace

std::unique_ptr<cudf::column> extractPointCoordinate(
    cudf::column_view const& geometry,
    bool extractY,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      geometry.type().id() == cudf::type_id::STRING,
      "geometry input must be STRING/VARBINARY");

  cudf::strings_column_view strings(geometry);
  auto const size = geometry.size();
  auto out = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::FLOAT64},
      size,
      cudf::mask_state::ALL_VALID,
      stream,
      mr);
  auto* outPtr = out->mutable_view().data<double>();
  auto outMask = static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());

  auto chars = strings.chars_begin(stream);
  auto offsets = strings.offsets().begin<cudf::size_type>();
  auto inNull = geometry.null_mask();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [chars,
       offsets,
       inNull,
       outPtr,
       outMask,
       extractY,
       invalidTypeFlag,
       nullCount = geometry.null_count()] __device__(cudf::size_type i) {
        if (nullCount > 0 && inNull != nullptr &&
            !cudf::bit_is_set(inNull, i)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        auto const start = offsets[i];
        auto const end = offsets[i + 1];
        double x = 0;
        double y = 0;
        if (!readPointXY(chars + start, end - start, x, y, invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        if (isEmptyPoint(x, y)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        outPtr[i] = extractY ? y : x;
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

std::unique_ptr<cudf::column> pointPointDistance(
    cudf::column_view const& left,
    cudf::column_view const& right,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      left.type().id() == cudf::type_id::STRING &&
          right.type().id() == cudf::type_id::STRING,
      "geometry inputs must be STRING/VARBINARY");
  CUDF_EXPECTS(left.size() == right.size(), "geometry size mismatch");

  cudf::strings_column_view leftStr(left);
  cudf::strings_column_view rightStr(right);
  auto const size = left.size();
  auto out = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::FLOAT64},
      size,
      cudf::mask_state::ALL_VALID,
      stream,
      mr);
  auto* outPtr = out->mutable_view().data<double>();
  auto outMask = static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());

  auto leftChars = leftStr.chars_begin(stream);
  auto rightChars = rightStr.chars_begin(stream);
  auto leftOffsets = leftStr.offsets().begin<cudf::size_type>();
  auto rightOffsets = rightStr.offsets().begin<cudf::size_type>();
  auto leftNull = left.null_mask();
  auto rightNull = right.null_mask();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [leftChars,
       rightChars,
       leftOffsets,
       rightOffsets,
       leftNull,
       rightNull,
       outPtr,
       outMask,
       invalidTypeFlag,
       leftNullCount = left.null_count(),
       rightNullCount = right.null_count()] __device__(cudf::size_type i) {
        if ((leftNullCount > 0 && leftNull != nullptr &&
             !cudf::bit_is_set(leftNull, i)) ||
            (rightNullCount > 0 && rightNull != nullptr &&
             !cudf::bit_is_set(rightNull, i))) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        auto const ls = leftOffsets[i];
        auto const le = leftOffsets[i + 1];
        auto const rs = rightOffsets[i];
        auto const re = rightOffsets[i + 1];
        if (!readPointXY(leftChars + ls, le - ls, x1, y1, invalidTypeFlag) ||
            !readPointXY(rightChars + rs, re - rs, x2, y2, invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        if (isEmptyPoint(x1, y1) || isEmptyPoint(x2, y2)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        double const dx = x1 - x2;
        double const dy = y1 - y2;
        outPtr[i] = sqrt(dx * dx + dy * dy);
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

std::unique_ptr<cudf::column> makePointGeometry(
    cudf::column_view const& x,
    cudf::column_view const& y,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      x.type().id() == cudf::type_id::FLOAT64 &&
          y.type().id() == cudf::type_id::FLOAT64,
      "ST_Point expects double coordinates");
  CUDF_EXPECTS(x.size() == y.size(), "ST_Point size mismatch");

  auto const size = x.size();
  // offsets: size+1 entries, each row is exactly kPointBlobSize bytes.
  auto offsetsCol = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      size + 1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* offsets = offsetsCol->mutable_view().data<cudf::size_type>();
  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size + 1,
      [offsets] __device__(cudf::size_type i) {
        offsets[i] = i * kPointBlobSize;
      });

  rmm::device_uvector<char> chars(
      static_cast<std::size_t>(size) * kPointBlobSize, stream, mr);
  auto* charsPtr = chars.data();
  auto const* xPtr = x.data<double>();
  auto const* yPtr = y.data<double>();
  auto xNull = x.null_mask();
  auto yNull = y.null_mask();

  // Output null mask: null if either input is null.
  auto [nullMask, nullCount] =
      cudf::bitmask_and(cudf::table_view{{x, y}}, stream, mr);

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [charsPtr,
       xPtr,
       yPtr,
       xNull,
       yNull,
       xNullCount = x.null_count(),
       yNullCount = y.null_count()] __device__(cudf::size_type i) {
        if ((xNullCount > 0 && xNull != nullptr &&
             !cudf::bit_is_set(xNull, i)) ||
            (yNullCount > 0 && yNull != nullptr &&
             !cudf::bit_is_set(yNull, i))) {
          return;
        }
        double xv = xPtr[i];
        double yv = yPtr[i];
        char* row = charsPtr + static_cast<std::size_t>(i) * kPointBlobSize;
        row[0] = static_cast<char>(kPointTag);
        auto const* xb = reinterpret_cast<char const*>(&xv);
        auto const* yb = reinterpret_cast<char const*>(&yv);
#pragma unroll
        for (int b = 0; b < 8; ++b) {
          row[1 + b] = xb[b];
          row[9 + b] = yb[b];
        }
      });

  return cudf::make_strings_column(
      size,
      std::move(offsetsCol),
      chars.release(),
      nullCount,
      std::move(nullMask));
}

std::unique_ptr<cudf::column> wkbPointToVeloxGeometry(
    cudf::column_view const& wkb,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      wkb.type().id() == cudf::type_id::STRING,
      "ST_GeomFromBinary expects VARBINARY/STRING WKB");

  cudf::strings_column_view strings(wkb);
  auto const size = wkb.size();

  auto offsetsCol = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      size + 1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* offsets = offsetsCol->mutable_view().data<cudf::size_type>();
  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size + 1,
      [offsets] __device__(cudf::size_type i) {
        offsets[i] = i * kPointBlobSize;
      });

  rmm::device_uvector<char> chars(
      static_cast<std::size_t>(size) * kPointBlobSize, stream, mr);
  auto* charsPtr = chars.data();

  auto inChars = strings.chars_begin(stream);
  auto inOffsets = strings.offsets().begin<cudf::size_type>();
  auto inNull = wkb.null_mask();

  auto [nullMaskBuf, nullCountHint] = [&]() {
    if (wkb.null_mask() != nullptr) {
      auto buf = cudf::copy_bitmask(wkb, stream, mr);
      return std::make_pair(std::move(buf), wkb.null_count());
    }
    return std::make_pair(
        cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr),
        0);
  }();
  auto* outMask = static_cast<cudf::bitmask_type*>(nullMaskBuf.data());

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [charsPtr,
       inChars,
       inOffsets,
       inNull,
       outMask,
       invalidTypeFlag,
       inNullCount = wkb.null_count()] __device__(cudf::size_type i) {
        if (inNullCount > 0 && inNull != nullptr &&
            !cudf::bit_is_set(inNull, i)) {
          return;
        }
        auto const start = inOffsets[i];
        auto const end = inOffsets[i + 1];
        char* row = charsPtr + static_cast<std::size_t>(i) * kPointBlobSize;
        if (!wkbPointToVeloxBlob(
                inChars + start, end - start, row, invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
        }
      });

  auto nullCount =
      cudf::null_count(static_cast<cudf::bitmask_type const*>(nullMaskBuf.data()), 0, size, stream);
  return cudf::make_strings_column(
      size,
      std::move(offsetsCol),
      chars.release(),
      nullCount,
      std::move(nullMaskBuf));
}

namespace {

template <typename T>
T readPod(char const*& p, char const* end) {
  if (p + sizeof(T) > end) {
    throw std::runtime_error("Truncated Velox geometry blob");
  }
  T v;
  std::memcpy(&v, p, sizeof(T));
  p += sizeof(T);
  return v;
}

} // namespace

bool parseVeloxPolygon(
    std::string_view geometry,
    std::vector<double>& xyOut,
    std::vector<int32_t>& partEndsOut) {
  xyOut.clear();
  partEndsOut.clear();
  if (geometry.empty()) {
    return false;
  }
  char const* p = geometry.data();
  char const* end = p + geometry.size();
  auto tag = static_cast<uint8_t>(readPod<char>(p, end));

  constexpr uint8_t kPolygonTag = 4;
  constexpr uint8_t kEnvelopeTag = 7;

  if (tag == kEnvelopeTag) {
    double xmin = readPod<double>(p, end);
    double ymin = readPod<double>(p, end);
    double xmax = readPod<double>(p, end);
    double ymax = readPod<double>(p, end);
    if (std::isnan(xmin) || std::isnan(ymin) || std::isnan(xmax) ||
        std::isnan(ymax)) {
      return false;
    }
    // Closed rectangle ring.
    xyOut = {
        xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin};
    partEndsOut = {5};
    return true;
  }

  if (tag != kPolygonTag) {
    return false;
  }
  // Skip Esri type + envelope.
  (void)readPod<int32_t>(p, end);
  (void)readPod<double>(p, end);
  (void)readPod<double>(p, end);
  (void)readPod<double>(p, end);
  (void)readPod<double>(p, end);

  int32_t numParts = readPod<int32_t>(p, end);
  int32_t numPoints = readPod<int32_t>(p, end);
  if (numParts <= 0 || numPoints <= 0) {
    return false;
  }
  std::vector<int32_t> starts(static_cast<size_t>(numParts));
  for (int32_t i = 0; i < numParts; ++i) {
    starts[static_cast<size_t>(i)] = readPod<int32_t>(p, end);
  }
  partEndsOut.resize(static_cast<size_t>(numParts));
  for (int32_t i = 0; i < numParts - 1; ++i) {
    partEndsOut[static_cast<size_t>(i)] = starts[static_cast<size_t>(i + 1)];
  }
  partEndsOut[static_cast<size_t>(numParts - 1)] = numPoints;

  xyOut.resize(static_cast<size_t>(numPoints) * 2);
  for (int32_t i = 0; i < numPoints; ++i) {
    xyOut[static_cast<size_t>(2 * i)] = readPod<double>(p, end);
    xyOut[static_cast<size_t>(2 * i + 1)] = readPod<double>(p, end);
  }
  return true;
}

std::unique_ptr<cudf::column> pointToConstantPolygonDistance(
    cudf::column_view const& points,
    DevicePolygonView polygon,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      points.type().id() == cudf::type_id::STRING,
      "points must be STRING/VARBINARY geometry");
  CUDF_EXPECTS(polygon.numParts > 0 && polygon.numPoints > 0, "empty polygon");
  CUDF_EXPECTS(
      polygon.xy != nullptr && polygon.partEnds != nullptr,
      "polygon device buffers required");

  cudf::strings_column_view strings(points);
  auto const size = points.size();
  auto out = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::FLOAT64},
      size,
      cudf::mask_state::ALL_VALID,
      stream,
      mr);
  auto* outPtr = out->mutable_view().data<double>();
  auto outMask =
      static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());

  auto chars = strings.chars_begin(stream);
  auto offsets = strings.offsets().begin<cudf::size_type>();
  auto inNull = points.null_mask();
  auto const* xy = polygon.xy;
  auto const* partEnds = polygon.partEnds;
  auto const numParts = polygon.numParts;

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [chars,
       offsets,
       inNull,
       outPtr,
       outMask,
       invalidTypeFlag,
       xy,
       partEnds,
       numParts,
       nullCount = points.null_count()] __device__(cudf::size_type i) {
        if (nullCount > 0 && inNull != nullptr &&
            !cudf::bit_is_set(inNull, i)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        double px = 0;
        double py = 0;
        auto const start = offsets[i];
        auto const end = offsets[i + 1];
        if (!readPointXY(chars + start, end - start, px, py, invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        if (isEmptyPoint(px, py)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }

        int32_t shellStart = 0;
        int32_t shellEnd = partEnds[0];
        bool const inShell = pointInRing(px, py, xy, shellStart, shellEnd);
        bool inHole = false;
        if (inShell) {
          for (int32_t p = 1; p < numParts; ++p) {
            int32_t hs = partEnds[p - 1];
            int32_t he = partEnds[p];
            if (pointInRing(px, py, xy, hs, he)) {
              inHole = true;
              break;
            }
          }
        }
        if (inShell && !inHole) {
          outPtr[i] = 0.0;
          return;
        }

        double best = INFINITY;
        for (int32_t p = 0; p < numParts; ++p) {
          int32_t ps = (p == 0) ? 0 : partEnds[p - 1];
          int32_t pe = partEnds[p];
          best = fmin(best, minDistToRing(px, py, xy, ps, pe));
        }
        outPtr[i] = best;
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

} // namespace facebook::velox::cudf_velox
