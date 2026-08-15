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
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <glog/logging.h>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

// Must match GeometrySerializationType in GeometryConstants.h.
constexpr uint8_t kPointTag = 0;
constexpr uint8_t kMultiPointTag = 1;
constexpr uint8_t kLineStringTag = 2;
constexpr uint8_t kMultiLineStringTag = 3;
constexpr uint8_t kPolygonTag = 4;
constexpr uint8_t kMultiPolygonTag = 5;
constexpr uint8_t kEnvelopeTag = 7;
constexpr int32_t kPointBlobSize = 17; // 1 + 8 + 8
constexpr int32_t kEsriPolygon = 5;
constexpr int32_t kEsriPolyline = 3;
constexpr int32_t kEmptyPolygonBlobSize = 45; // tag+esri+envelope+numParts+numPoints
constexpr int32_t kEmptyLineStringBlobSize = 45;
constexpr uint32_t kWkbSridFlag = 0x20000000u;

// Forward decls — used by polygonsIntersectBlob before their definitions.
__device__ inline bool readEnvelopeFromBlob(
    char const* data,
    cudf::size_type len,
    double& minX,
    double& minY,
    double& maxX,
    double& maxY,
    int32_t* invalidTypeFlag);
__device__ inline bool envelopesIntersect(
    double aMinX,
    double aMinY,
    double aMaxX,
    double aMaxY,
    double bMinX,
    double bMinY,
    double bMaxX,
    double bMaxY);

__device__ inline int32_t veloxLineStringBlobSize(int32_t numPoints) {
  // tag + esri + envelope + numParts + numPoints + partStarts[1] + xy
  return kEmptyLineStringBlobSize + 4 + 16 * numPoints;
}

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

__device__ inline int32_t readI32Native(char const* p) {
  int32_t v = 0;
  auto* outb = reinterpret_cast<char*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    outb[i] = p[i];
  }
  return v;
}

__device__ inline double readF64Native(char const* p) {
  double v = 0;
  auto* outb = reinterpret_cast<char*>(&v);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    outb[i] = p[i];
  }
  return v;
}

__device__ inline void writeI32Native(char* p, int32_t v) {
  auto const* b = reinterpret_cast<char const*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p[i] = b[i];
  }
}

__device__ inline void writeF64Native(char* p, double v) {
  auto const* b = reinterpret_cast<char const*>(&v);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    p[i] = b[i];
  }
}

__device__ inline double xyBytesX(char const* xy, int32_t i) {
  return readF64Native(xy + static_cast<std::size_t>(i) * 16);
}

__device__ inline double xyBytesY(char const* xy, int32_t i) {
  return readF64Native(xy + static_cast<std::size_t>(i) * 16 + 8);
}

__device__ inline bool pointInRingBytes(
    double px,
    double py,
    char const* xy,
    int32_t partStart,
    int32_t partEnd) {
  if (partEnd - partStart < 3) {
    return false;
  }
  bool inside = false;
  for (int32_t i = partStart, j = partEnd - 1; i < partEnd; j = i++) {
    double const xi = xyBytesX(xy, i);
    double const yi = xyBytesY(xy, i);
    double const xj = xyBytesX(xy, j);
    double const yj = xyBytesY(xy, j);
    bool const intersect = ((yi > py) != (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi + 0.0) + xi);
    if (intersect) {
      inside = !inside;
    }
  }
  return inside;
}

/// Velox GeometrySerde: exterior rings are clockwise, holes counter-clockwise.
__device__ inline bool ringIsClockwiseBytes(
    char const* xy,
    int32_t partStart,
    int32_t partEnd) {
  if (partEnd - partStart < 3) {
    return true;
  }
  double sum = 0.0;
  for (int32_t i = partStart; i < partEnd - 1; ++i) {
    double const x1 = xyBytesX(xy, i);
    double const y1 = xyBytesY(xy, i);
    double const x2 = xyBytesX(xy, i + 1);
    double const y2 = xyBytesY(xy, i + 1);
    sum += (x2 - x1) * (y2 + y1);
  }
  // Close ring.
  double const x1 = xyBytesX(xy, partEnd - 1);
  double const y1 = xyBytesY(xy, partEnd - 1);
  double const x2 = xyBytesX(xy, partStart);
  double const y2 = xyBytesY(xy, partStart);
  sum += (x2 - x1) * (y2 + y1);
  return sum > 0.0;
}

__device__ inline void reverseRingBytes(
    char* xy,
    int32_t partStart,
    int32_t partEnd) {
  int32_t a = partStart;
  int32_t b = partEnd - 1;
  while (a < b) {
    double const ax = xyBytesX(xy, a);
    double const ay = xyBytesY(xy, a);
    double const bx = xyBytesX(xy, b);
    double const by = xyBytesY(xy, b);
    writeF64Native(xy + static_cast<std::size_t>(a) * 16, bx);
    writeF64Native(xy + static_cast<std::size_t>(a) * 16 + 8, by);
    writeF64Native(xy + static_cast<std::size_t>(b) * 16, ax);
    writeF64Native(xy + static_cast<std::size_t>(b) * 16 + 8, ay);
    ++a;
    --b;
  }
}

__device__ inline double minDistToRingBytes(
    double px,
    double py,
    char const* xy,
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
        xyBytesX(xy, i),
        xyBytesY(xy, i),
        xyBytesX(xy, i + 1),
        xyBytesY(xy, i + 1));
    best = fmin(best, d);
  }
  int32_t const last = partEnd - 1;
  double const dClose = distPointSegment(
      px,
      py,
      xyBytesX(xy, last),
      xyBytesY(xy, last),
      xyBytesX(xy, partStart),
      xyBytesY(xy, partStart));
  return fmin(best, dClose);
}

/// Distance from point to Velox POLYGON/ENVELOPE blob (unaligned-safe).
__device__ inline bool distPointToPolygonBlob(
    double px,
    double py,
    char const* data,
    cudf::size_type len,
    double& outDist,
    int32_t* invalidTypeFlag) {
  if (len < 1) {
    markInvalid(invalidTypeFlag);
    return false;
  }
  uint8_t const tag = static_cast<uint8_t>(data[0]);

  char envXy[80]; // 5 points * 16
  char const* xy = nullptr;
  int32_t numParts = 0;
  int32_t numPoints = 0;
  bool envelope = false;

  if (tag == kEnvelopeTag) {
    if (len < 1 + 32) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    double const xmin = readF64Native(data + 1);
    double const ymin = readF64Native(data + 9);
    double const xmax = readF64Native(data + 17);
    double const ymax = readF64Native(data + 25);
    double coords[10] = {
        xmin,
        ymin,
        xmax,
        ymin,
        xmax,
        ymax,
        xmin,
        ymax,
        xmin,
        ymin};
#pragma unroll
    for (int i = 0; i < 10; ++i) {
      writeF64Native(envXy + i * 8, coords[i]);
    }
    xy = envXy;
    numParts = 1;
    numPoints = 5;
    envelope = true;
  } else if (tag == kPolygonTag) {
    if (len < kEmptyPolygonBlobSize) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    numParts = readI32Native(data + 37);
    numPoints = readI32Native(data + 41);
    if (numParts <= 0 || numPoints <= 0) {
      return false; // empty → null
    }
    std::size_t const partsBytes =
        static_cast<std::size_t>(numParts) * sizeof(int32_t);
    std::size_t const xyBytesNeeded =
        static_cast<std::size_t>(numPoints) * 16;
    if (static_cast<std::size_t>(len) <
        static_cast<std::size_t>(kEmptyPolygonBlobSize) + partsBytes +
            xyBytesNeeded) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    xy = data + kEmptyPolygonBlobSize +
        static_cast<std::size_t>(numParts) * 4;
  } else {
    markInvalid(invalidTypeFlag);
    return false;
  }

  // Compute part starts/ends without lambdas (device-friendly).
  int32_t shellStart = 0;
  int32_t shellEnd = numPoints;
  if (!envelope) {
    shellStart = readI32Native(data + kEmptyPolygonBlobSize);
    shellEnd = (numParts > 1)
        ? readI32Native(data + kEmptyPolygonBlobSize + 4)
        : numPoints;
  }

  bool const inShell =
      pointInRingBytes(px, py, xy, shellStart, shellEnd);
  bool inHole = false;
  if (inShell && !envelope) {
    for (int32_t p = 1; p < numParts; ++p) {
      int32_t const hs = readI32Native(
          data + kEmptyPolygonBlobSize + static_cast<std::size_t>(p) * 4);
      int32_t const he = (p + 1 < numParts)
          ? readI32Native(
                data + kEmptyPolygonBlobSize +
                static_cast<std::size_t>(p + 1) * 4)
          : numPoints;
      if (pointInRingBytes(px, py, xy, hs, he)) {
        inHole = true;
        break;
      }
    }
  }
  if (inShell && !inHole) {
    outDist = 0.0;
    return true;
  }

  double best = INFINITY;
  for (int32_t p = 0; p < numParts; ++p) {
    int32_t ps = 0;
    int32_t pe = numPoints;
    if (!envelope) {
      ps = readI32Native(
          data + kEmptyPolygonBlobSize + static_cast<std::size_t>(p) * 4);
      pe = (p + 1 < numParts)
          ? readI32Native(
                data + kEmptyPolygonBlobSize +
                static_cast<std::size_t>(p + 1) * 4)
          : numPoints;
    }
    best = fmin(best, minDistToRingBytes(px, py, xy, ps, pe));
  }
  outDist = best;
  return true;
}

/// Point-in-polygon for Velox POLYGON/ENVELOPE blob. Returns false in
/// *outValid when the blob is empty (null result); marks *invalidTypeFlag
/// for unsupported types / corrupt blobs.
__device__ inline bool pointInPolygonBlob(
    double px,
    double py,
    char const* data,
    cudf::size_type len,
    bool& outInside,
    int32_t* invalidTypeFlag) {
  outInside = false;
  if (len < 1) {
    markInvalid(invalidTypeFlag);
    return false;
  }
  uint8_t const tag = static_cast<uint8_t>(data[0]);

  char envXy[80];
  char const* xy = nullptr;
  int32_t numParts = 0;
  int32_t numPoints = 0;
  bool envelope = false;

  if (tag == kEnvelopeTag) {
    if (len < 1 + 32) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    double const xmin = readF64Native(data + 1);
    double const ymin = readF64Native(data + 9);
    double const xmax = readF64Native(data + 17);
    double const ymax = readF64Native(data + 25);
    double coords[10] = {
        xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin};
#pragma unroll
    for (int i = 0; i < 10; ++i) {
      writeF64Native(envXy + i * 8, coords[i]);
    }
    xy = envXy;
    numParts = 1;
    numPoints = 5;
    envelope = true;
  } else if (tag == kPolygonTag || tag == kMultiPolygonTag) {
    if (len < kEmptyPolygonBlobSize) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    // cuSpatial-style coarse reject: Velox polygons embed xmin/ymin/xmax/ymax
    // at byte 5. Envelope-grid candidates often miss the true polygon (esp.
    // continent-scale zones); bail before any ring traversal.
    {
      double const xmin = readF64Native(data + 5);
      double const ymin = readF64Native(data + 13);
      double const xmax = readF64Native(data + 21);
      double const ymax = readF64Native(data + 29);
      if (!(px >= xmin && px <= xmax && py >= ymin && py <= ymax)) {
        outInside = false;
        return true;
      }
    }
    numParts = readI32Native(data + 37);
    numPoints = readI32Native(data + 41);
    if (numParts <= 0 || numPoints <= 0) {
      return false; // empty → null
    }
    std::size_t const partsBytes =
        static_cast<std::size_t>(numParts) * sizeof(int32_t);
    std::size_t const xyBytesNeeded =
        static_cast<std::size_t>(numPoints) * 16;
    if (static_cast<std::size_t>(len) <
        static_cast<std::size_t>(kEmptyPolygonBlobSize) + partsBytes +
            xyBytesNeeded) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    // Validate part offsets so corrupt blobs cannot OOB in pointInRingBytes.
    int32_t prevPs = 0;
    for (int32_t p = 0; p < numParts; ++p) {
      int32_t const ps = readI32Native(
          data + kEmptyPolygonBlobSize + static_cast<std::size_t>(p) * 4);
      if (ps < 0 || ps > numPoints || (p > 0 && ps < prevPs)) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      prevPs = ps;
    }
    xy = data + kEmptyPolygonBlobSize +
        static_cast<std::size_t>(numParts) * 4;
  } else {
    markInvalid(invalidTypeFlag);
    return false;
  }

  if (envelope) {
    outInside = pointInRingBytes(px, py, xy, 0, numPoints);
    return true;
  }

  // POLYGON: part 0 = shell, parts 1.. = holes (WKB / single-polygon layout).
  if (tag == kPolygonTag) {
    int32_t shellStart = readI32Native(data + kEmptyPolygonBlobSize);
    int32_t shellEnd = (numParts > 1)
        ? readI32Native(data + kEmptyPolygonBlobSize + 4)
        : numPoints;
    bool const inShell = pointInRingBytes(px, py, xy, shellStart, shellEnd);
    if (!inShell) {
      outInside = false;
      return true;
    }
    for (int32_t p = 1; p < numParts; ++p) {
      int32_t const hs = readI32Native(
          data + kEmptyPolygonBlobSize + static_cast<std::size_t>(p) * 4);
      int32_t const he = (p + 1 < numParts)
          ? readI32Native(
                data + kEmptyPolygonBlobSize +
                static_cast<std::size_t>(p + 1) * 4)
          : numPoints;
      if (pointInRingBytes(px, py, xy, hs, he)) {
        outInside = false;
        return true;
      }
    }
    outInside = true;
    return true;
  }

  // MULTI_POLYGON: shells are clockwise, holes counter-clockwise (Velox serde).
  for (int32_t p = 0; p < numParts; ++p) {
    int32_t const ps = readI32Native(
        data + kEmptyPolygonBlobSize + static_cast<std::size_t>(p) * 4);
    int32_t const pe = (p + 1 < numParts)
        ? readI32Native(
              data + kEmptyPolygonBlobSize +
              static_cast<std::size_t>(p + 1) * 4)
        : numPoints;
    if (!ringIsClockwiseBytes(xy, ps, pe)) {
      continue; // hole — handled under its shell
    }
    if (!pointInRingBytes(px, py, xy, ps, pe)) {
      continue;
    }
    bool inHole = false;
    for (int32_t h = p + 1; h < numParts; ++h) {
      int32_t const hs = readI32Native(
          data + kEmptyPolygonBlobSize + static_cast<std::size_t>(h) * 4);
      int32_t const he = (h + 1 < numParts)
          ? readI32Native(
                data + kEmptyPolygonBlobSize +
                static_cast<std::size_t>(h + 1) * 4)
          : numPoints;
      if (ringIsClockwiseBytes(xy, hs, he)) {
        break; // next shell
      }
      if (pointInRingBytes(px, py, xy, hs, he)) {
        inHole = true;
        break;
      }
    }
    if (!inHole) {
      outInside = true;
      return true;
    }
  }
  outInside = false;
  return true;
}

/// Proper segment intersection (shared interior point), excluding pure
/// endpoint-touch-only cases handled via vertex-in-poly separately.
__device__ inline bool segmentsIntersectProper(
    double ax,
    double ay,
    double bx,
    double by,
    double cx,
    double cy,
    double dx,
    double dy) {
  auto cross = [](double ux, double uy, double vx, double vy) {
    return ux * vy - uy * vx;
  };
  double const abx = bx - ax;
  double const aby = by - ay;
  double const acx = cx - ax;
  double const acy = cy - ay;
  double const adx = dx - ax;
  double const ady = dy - ay;
  double const cdx = dx - cx;
  double const cdy = dy - cy;
  double const d1 = cross(abx, aby, acx, acy);
  double const d2 = cross(abx, aby, adx, ady);
  double const d3 = cross(cdx, cdy, ax - cx, ay - cy);
  double const d4 = cross(cdx, cdy, bx - cx, by - cy);
  auto sgn = [](double v) {
    return (v > 0.0) - (v < 0.0);
  };
  return sgn(d1) != sgn(d2) && sgn(d3) != sgn(d4) && d1 != 0.0 && d2 != 0.0 &&
      d3 != 0.0 && d4 != 0.0;
}

__device__ inline bool parsePolygonRings(
    char const* data,
    cudf::size_type len,
    char const*& xyOut,
    char* envScratch,
    int32_t& numParts,
    int32_t& numPoints,
    bool& envelope,
    int32_t* invalidTypeFlag) {
  if (len < 1) {
    markInvalid(invalidTypeFlag);
    return false;
  }
  uint8_t const tag = static_cast<uint8_t>(data[0]);
  envelope = false;
  if (tag == kEnvelopeTag) {
    if (len < 1 + 32) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    double const xmin = readF64Native(data + 1);
    double const ymin = readF64Native(data + 9);
    double const xmax = readF64Native(data + 17);
    double const ymax = readF64Native(data + 25);
    double coords[10] = {
        xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin};
#pragma unroll
    for (int i = 0; i < 10; ++i) {
      writeF64Native(envScratch + i * 8, coords[i]);
    }
    xyOut = envScratch;
    numParts = 1;
    numPoints = 5;
    envelope = true;
    return true;
  }
  if (tag != kPolygonTag && tag != kMultiPolygonTag) {
    markInvalid(invalidTypeFlag);
    return false;
  }
  if (len < kEmptyPolygonBlobSize) {
    markInvalid(invalidTypeFlag);
    return false;
  }
  numParts = readI32Native(data + 37);
  numPoints = readI32Native(data + 41);
  if (numParts <= 0 || numPoints <= 0) {
    return false;
  }
  std::size_t const partsBytes =
      static_cast<std::size_t>(numParts) * sizeof(int32_t);
  std::size_t const xyBytesNeeded = static_cast<std::size_t>(numPoints) * 16;
  if (static_cast<std::size_t>(len) <
      static_cast<std::size_t>(kEmptyPolygonBlobSize) + partsBytes +
          xyBytesNeeded) {
    markInvalid(invalidTypeFlag);
    return false;
  }
  xyOut = data + kEmptyPolygonBlobSize + static_cast<std::size_t>(numParts) * 4;
  return true;
}

__device__ inline void ringBounds(
    char const* data,
    bool envelope,
    int32_t part,
    int32_t numParts,
    int32_t numPoints,
    int32_t& ps,
    int32_t& pe) {
  if (envelope) {
    ps = 0;
    pe = numPoints;
    return;
  }
  ps = readI32Native(
      data + kEmptyPolygonBlobSize + static_cast<std::size_t>(part) * 4);
  pe = (part + 1 < numParts)
      ? readI32Native(
            data + kEmptyPolygonBlobSize +
            static_cast<std::size_t>(part + 1) * 4)
      : numPoints;
}

// Larger dimension of a polygon blob's embedded whole-geometry bbox; 0 for
// non-polygon or truncated blobs. Used to decide which geometries are worth
// decomposing into rings.
__device__ inline double blobBboxSpan(char const* data, cudf::size_type len) {
  if (len < 5 + 32) {
    return 0.0;
  }
  uint8_t const tag = static_cast<uint8_t>(data[0]);
  if (tag != kPolygonTag && tag != kMultiPolygonTag) {
    return 0.0;
  }
  double const dx = readF64Native(data + 21) - readF64Native(data + 5);
  double const dy = readF64Native(data + 29) - readF64Native(data + 13);
  return fmax(dx, dy);
}

// Number of index entries a blob contributes to a per-part envelope index:
//   0 for empty/too-short blobs,
//   numRings for polygon/multipolygon when 0 < numRings <= maxParts and the
//     whole-geometry bbox is at least minSplitSpan across,
//   1 otherwise (points, envelopes, linestrings, ordinary-sized polygons, or
//     ring counts above the cap, which keep the whole-geometry bounding box).
// Only outsized geometries are decomposed: splitting every polygon inflates the
// index by the total ring count (~1.9B for SF100 zones), and the grid query
// allocates O(index) 64-bit scratch, so an unfiltered split OOMs regardless of
// batch size. The antimeridian/island multipolygons that motivate the split are
// exactly the ones with globe-spanning bboxes.
// This must stay in exact agreement with the per-part vs whole-bbox decision in
// extractGeometryPartEnvelopes' second pass.
__device__ inline int32_t blobIndexPartCount(
    char const* data,
    cudf::size_type len,
    int32_t maxParts,
    double minSplitSpan) {
  if (len < 1) {
    return 0;
  }
  uint8_t const tag = static_cast<uint8_t>(data[0]);
  if (tag == kPolygonTag || tag == kMultiPolygonTag) {
    if (len < kEmptyPolygonBlobSize) {
      return 0;
    }
    int32_t const numParts = readI32Native(data + 37);
    int32_t const numPoints = readI32Native(data + 41);
    if (numParts <= 0 || numPoints <= 0) {
      return 1; // degenerate → single whole-bbox entry (will null out)
    }
    if (numParts > maxParts) {
      return 1; // too many rings → keep the coarse whole-geometry bbox
    }
    if (blobBboxSpan(data, len) < minSplitSpan) {
      return 1; // compact geometry → whole bbox is already tight
    }
    return numParts;
  }
  return 1; // point / envelope / linestring → single whole-bbox entry
}

// True when the second pass should compute a per-ring bbox for this blob (vs.
// the single whole-geometry bbox). Mirrors blobIndexPartCount.
__device__ inline bool blobUsesPerRing(
    char const* data,
    cudf::size_type len,
    int32_t maxParts,
    double minSplitSpan,
    int32_t& numParts,
    int32_t& numPoints) {
  numParts = 0;
  numPoints = 0;
  if (len < 1) {
    return false;
  }
  uint8_t const tag = static_cast<uint8_t>(data[0]);
  if (tag != kPolygonTag && tag != kMultiPolygonTag) {
    return false;
  }
  if (len < kEmptyPolygonBlobSize) {
    return false;
  }
  numParts = readI32Native(data + 37);
  numPoints = readI32Native(data + 41);
  if (numParts <= 0 || numPoints <= 0 || numParts > maxParts) {
    return false;
  }
  return blobBboxSpan(data, len) >= minSplitSpan;
}

/// Polygon/envelope intersects polygon/envelope (SpatialBench Q6).
__device__ inline bool polygonsIntersectBlob(
    char const* aData,
    cudf::size_type aLen,
    char const* bData,
    cudf::size_type bLen,
    bool& outIntersect,
    int32_t* invalidTypeFlag) {
  outIntersect = false;
  double aMinX = 0, aMinY = 0, aMaxX = 0, aMaxY = 0;
  double bMinX = 0, bMinY = 0, bMaxX = 0, bMaxY = 0;
  if (!readEnvelopeFromBlob(
          aData, aLen, aMinX, aMinY, aMaxX, aMaxY, invalidTypeFlag) ||
      !readEnvelopeFromBlob(
          bData, bLen, bMinX, bMinY, bMaxX, bMaxY, invalidTypeFlag)) {
    return false;
  }
  if (!envelopesIntersect(
          aMinX, aMinY, aMaxX, aMaxY, bMinX, bMinY, bMaxX, bMaxY)) {
    outIntersect = false;
    return true;
  }

  char aEnv[80];
  char bEnv[80];
  char const* aXy = nullptr;
  char const* bXy = nullptr;
  int32_t aParts = 0, aPoints = 0, bParts = 0, bPoints = 0;
  bool aEnvTag = false, bEnvTag = false;
  if (!parsePolygonRings(
          aData,
          aLen,
          aXy,
          aEnv,
          aParts,
          aPoints,
          aEnvTag,
          invalidTypeFlag) ||
      !parsePolygonRings(
          bData,
          bLen,
          bXy,
          bEnv,
          bParts,
          bPoints,
          bEnvTag,
          invalidTypeFlag)) {
    return false;
  }

  // Any vertex of A inside B (or vice versa).
  for (int32_t i = 0; i < aPoints; ++i) {
    bool inside = false;
    if (!pointInPolygonBlob(
            xyBytesX(aXy, i),
            xyBytesY(aXy, i),
            bData,
            bLen,
            inside,
            invalidTypeFlag)) {
      return false;
    }
    if (inside) {
      outIntersect = true;
      return true;
    }
  }
  for (int32_t i = 0; i < bPoints; ++i) {
    bool inside = false;
    if (!pointInPolygonBlob(
            xyBytesX(bXy, i),
            xyBytesY(bXy, i),
            aData,
            aLen,
            inside,
            invalidTypeFlag)) {
      return false;
    }
    if (inside) {
      outIntersect = true;
      return true;
    }
  }

  // Shell edge crossings (part 0). Holes are covered by vertex-in tests for
  // typical SpatialBench zones; edge×edge on shells catches overlapping rings.
  int32_t aPs = 0, aPe = 0, bPs = 0, bPe = 0;
  ringBounds(aData, aEnvTag, 0, aParts, aPoints, aPs, aPe);
  ringBounds(bData, bEnvTag, 0, bParts, bPoints, bPs, bPe);
  auto nextIdx = [](int32_t i, int32_t start, int32_t end) {
    return (i + 1 < end) ? (i + 1) : start;
  };
  for (int32_t i = aPs; i < aPe; ++i) {
    int32_t const in = nextIdx(i, aPs, aPe);
    double const ax = xyBytesX(aXy, i);
    double const ay = xyBytesY(aXy, i);
    double const bx = xyBytesX(aXy, in);
    double const by = xyBytesY(aXy, in);
    for (int32_t j = bPs; j < bPe; ++j) {
      int32_t const jn = nextIdx(j, bPs, bPe);
      if (segmentsIntersectProper(
              ax,
              ay,
              bx,
              by,
              xyBytesX(bXy, j),
              xyBytesY(bXy, j),
              xyBytesX(bXy, jn),
              xyBytesY(bXy, jn))) {
        outIntersect = true;
        return true;
      }
    }
  }
  outIntersect = false;
  return true;
}

// --- Sutherland–Hodgman polygon ∩ polygon (SpatialBench Q9 buildings) ------
// Buildings are simple convex footprints (~5–8 verts). Clip polygon must be
// convex; we normalize both rings to CCW before clipping, then emit CW for
// Velox Esri exterior convention.
constexpr int32_t kMaxClipRingVerts = 64;

__device__ inline bool isLeftCCW(
    double ax,
    double ay,
    double bx,
    double by,
    double px,
    double py) {
  return (bx - ax) * (py - ay) - (by - ay) * (px - ax) > 0.0;
}

__device__ inline void edgeIntersect(
    double ax,
    double ay,
    double bx,
    double by,
    double cx,
    double cy,
    double dx,
    double dy,
    double& ox,
    double& oy) {
  // Intersection of infinite lines AB and CD.
  double const a1 = by - ay;
  double const b1 = ax - bx;
  double const c1 = a1 * ax + b1 * ay;
  double const a2 = dy - cy;
  double const b2 = cx - dx;
  double const c2 = a2 * cx + b2 * cy;
  double const det = a1 * b2 - a2 * b1;
  if (fabs(det) < 1e-18) {
    ox = cx;
    oy = cy;
    return;
  }
  ox = (b2 * c1 - b1 * c2) / det;
  oy = (a1 * c2 - a2 * c1) / det;
}

/// Load exterior ring (part 0) into interleaved xy[2*i]/xy[2*i+1]. Returns
/// vertex count excluding a duplicate closing vertex if present. 0 on failure.
__device__ inline int32_t loadExteriorRingXY(
    char const* data,
    cudf::size_type len,
    double* xyOut,
    int32_t maxVerts,
    int32_t* invalidTypeFlag) {
  if (len < 1) {
    return 0;
  }
  uint8_t const tag = static_cast<uint8_t>(data[0]);
  char envScratch[80];
  char const* xy = nullptr;
  int32_t numParts = 0;
  int32_t numPoints = 0;
  bool envelope = false;
  if (tag == kEnvelopeTag) {
    if (len < 1 + 32) {
      markInvalid(invalidTypeFlag);
      return 0;
    }
    double const xmin = readF64Native(data + 1);
    double const ymin = readF64Native(data + 9);
    double const xmax = readF64Native(data + 17);
    double const ymax = readF64Native(data + 25);
    double coords[10] = {
        xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin};
#pragma unroll
    for (int j = 0; j < 10; ++j) {
      writeF64Native(envScratch + j * 8, coords[j]);
    }
    xy = envScratch;
    numParts = 1;
    numPoints = 5;
    envelope = true;
  } else if (tag == kPolygonTag || tag == kMultiPolygonTag) {
    if (len < kEmptyPolygonBlobSize) {
      markInvalid(invalidTypeFlag);
      return 0;
    }
    numParts = readI32Native(data + 37);
    numPoints = readI32Native(data + 41);
    if (numParts <= 0 || numPoints <= 0) {
      return 0;
    }
    std::size_t const partsBytes =
        static_cast<std::size_t>(numParts) * sizeof(int32_t);
    std::size_t const xyBytes = static_cast<std::size_t>(numPoints) * 16;
    if (static_cast<std::size_t>(len) <
        static_cast<std::size_t>(kEmptyPolygonBlobSize) + partsBytes + xyBytes) {
      markInvalid(invalidTypeFlag);
      return 0;
    }
    xy = data + kEmptyPolygonBlobSize + partsBytes;
  } else {
    markInvalid(invalidTypeFlag);
    return 0;
  }

  int32_t ps = 0;
  int32_t pe = 0;
  if (envelope) {
    ps = 0;
    pe = numPoints;
  } else {
    ps = readI32Native(data + kEmptyPolygonBlobSize);
    pe = (numParts > 1)
        ? readI32Native(data + kEmptyPolygonBlobSize + 4)
        : numPoints;
  }
  int32_t n = pe - ps;
  if (n < 3) {
    return 0;
  }
  // Drop duplicate closing vertex.
  if (xyBytesX(xy, pe - 1) == xyBytesX(xy, ps) &&
      xyBytesY(xy, pe - 1) == xyBytesY(xy, ps)) {
    --n;
  }
  if (n < 3 || n > maxVerts) {
    if (n > maxVerts) {
      markInvalid(invalidTypeFlag);
    }
    return 0;
  }
  for (int32_t i = 0; i < n; ++i) {
    xyOut[2 * i] = xyBytesX(xy, ps + i);
    xyOut[2 * i + 1] = xyBytesY(xy, ps + i);
  }
  // Ensure CCW for SH (positive isLeft).
  double sum = 0.0;
  for (int32_t i = 0; i < n; ++i) {
    int32_t const j = (i + 1) % n;
    sum += (xyOut[2 * j] - xyOut[2 * i]) * (xyOut[2 * j + 1] + xyOut[2 * i + 1]);
  }
  if (sum > 0.0) { // CW under this convention → reverse to CCW
    for (int32_t a = 0, b = n - 1; a < b; ++a, --b) {
      double const tx = xyOut[2 * a];
      double const ty = xyOut[2 * a + 1];
      xyOut[2 * a] = xyOut[2 * b];
      xyOut[2 * a + 1] = xyOut[2 * b + 1];
      xyOut[2 * b] = tx;
      xyOut[2 * b + 1] = ty;
    }
  }
  return n;
}

/// Clip subject (CCW) against convex clip (CCW). Writes CCW result without
/// closing vertex. Returns vertex count (0 if empty / too large).
__device__ inline int32_t sutherlandHodgman(
    double const* subj,
    int32_t nSubj,
    double const* clip,
    int32_t nClip,
    double* out,
    int32_t maxOut) {
  if (nSubj < 3 || nClip < 3) {
    return 0;
  }
  double bufA[2 * kMaxClipRingVerts];
  double bufB[2 * kMaxClipRingVerts];
  for (int32_t i = 0; i < nSubj; ++i) {
    bufA[2 * i] = subj[2 * i];
    bufA[2 * i + 1] = subj[2 * i + 1];
  }
  int32_t nIn = nSubj;
  double* input = bufA;
  double* output = bufB;

  for (int32_t c = 0; c < nClip; ++c) {
    int32_t const cn = (c + 1) % nClip;
    double const ax = clip[2 * c];
    double const ay = clip[2 * c + 1];
    double const bx = clip[2 * cn];
    double const by = clip[2 * cn + 1];
    int32_t nOut = 0;
    if (nIn == 0) {
      return 0;
    }
    for (int32_t i = 0; i < nIn; ++i) {
      int32_t const j = (i + 1) % nIn;
      double const sx = input[2 * i];
      double const sy = input[2 * i + 1];
      double const ex = input[2 * j];
      double const ey = input[2 * j + 1];
      bool const sIn = isLeftCCW(ax, ay, bx, by, sx, sy);
      bool const eIn = isLeftCCW(ax, ay, bx, by, ex, ey);
      if (eIn) {
        if (!sIn) {
          if (nOut >= maxOut) {
            return 0;
          }
          edgeIntersect(sx, sy, ex, ey, ax, ay, bx, by, output[2 * nOut], output[2 * nOut + 1]);
          ++nOut;
        }
        if (nOut >= maxOut) {
          return 0;
        }
        output[2 * nOut] = ex;
        output[2 * nOut + 1] = ey;
        ++nOut;
      } else if (sIn) {
        if (nOut >= maxOut) {
          return 0;
        }
        edgeIntersect(sx, sy, ex, ey, ax, ay, bx, by, output[2 * nOut], output[2 * nOut + 1]);
        ++nOut;
      }
    }
    // Swap buffers.
    double* tmp = input;
    input = output;
    output = tmp;
    nIn = nOut;
  }
  if (nIn < 3 || nIn > maxOut) {
    return 0;
  }
  for (int32_t i = 0; i < nIn; ++i) {
    out[2 * i] = input[2 * i];
    out[2 * i + 1] = input[2 * i + 1];
  }
  return nIn;
}

__device__ inline void writeEmptyPolygonBlob(char* out) {
  out[0] = static_cast<char>(kPolygonTag);
  writeI32Native(out + 1, kEsriPolygon);
  double const nan = NAN;
  writeF64Native(out + 5, nan);
  writeF64Native(out + 13, nan);
  writeF64Native(out + 21, nan);
  writeF64Native(out + 29, nan);
  writeI32Native(out + 37, 0);
  writeI32Native(out + 41, 0);
}

/// Write a single-ring CW Esri polygon (closed: n+1 points with duplicate).
__device__ inline void writeSingleRingPolygonBlob(
    char* out,
    double const* xyCCW,
    int32_t n) {
  // Emit CW (reverse CCW) with closing vertex.
  int32_t const numPoints = n + 1;
  out[0] = static_cast<char>(kPolygonTag);
  writeI32Native(out + 1, kEsriPolygon);
  writeI32Native(out + 37, 1);
  writeI32Native(out + 41, numPoints);
  writeI32Native(out + kEmptyPolygonBlobSize, 0);
  char* xyPtr = out + kEmptyPolygonBlobSize + 4;
  double xmin = INFINITY, ymin = INFINITY, xmax = -INFINITY, ymax = -INFINITY;
  for (int32_t i = 0; i < n; ++i) {
    // Reverse: CCW[i] → CW index (n-1-i)
    int32_t const src = n - 1 - i;
    double const x = xyCCW[2 * src];
    double const y = xyCCW[2 * src + 1];
    writeF64Native(xyPtr + static_cast<std::size_t>(i) * 16, x);
    writeF64Native(xyPtr + static_cast<std::size_t>(i) * 16 + 8, y);
    xmin = fmin(xmin, x);
    ymin = fmin(ymin, y);
    xmax = fmax(xmax, x);
    ymax = fmax(ymax, y);
  }
  // Close with first CW point (= last CCW vertex).
  double const x0 = xyCCW[2 * (n - 1)];
  double const y0 = xyCCW[2 * (n - 1) + 1];
  writeF64Native(xyPtr + static_cast<std::size_t>(n) * 16, x0);
  writeF64Native(xyPtr + static_cast<std::size_t>(n) * 16 + 8, y0);
  writeF64Native(out + 5, xmin);
  writeF64Native(out + 13, ymin);
  writeF64Native(out + 21, xmax);
  writeF64Native(out + 29, ymax);
}

__device__ inline int32_t veloxPolygonBlobSize(int32_t numParts, int32_t numPoints) {
  return kEmptyPolygonBlobSize + 4 * numParts + 16 * numPoints;
}

/// Clip two polygon blobs; return output blob size (empty=45) or 0 if null.
__device__ inline int32_t clipPolygonsBlobSize(
    char const* aData,
    cudf::size_type aLen,
    char const* bData,
    cudf::size_type bLen,
    int32_t* invalidTypeFlag) {
  double aMinX, aMinY, aMaxX, aMaxY, bMinX, bMinY, bMaxX, bMaxY;
  if (!readEnvelopeFromBlob(
          aData, aLen, aMinX, aMinY, aMaxX, aMaxY, invalidTypeFlag) ||
      !readEnvelopeFromBlob(
          bData, bLen, bMinX, bMinY, bMaxX, bMaxY, invalidTypeFlag)) {
    return 0; // null
  }
  if (!envelopesIntersect(
          aMinX, aMinY, aMaxX, aMaxY, bMinX, bMinY, bMaxX, bMaxY)) {
    return kEmptyPolygonBlobSize;
  }
  double subj[2 * kMaxClipRingVerts];
  double clip[2 * kMaxClipRingVerts];
  double out[2 * kMaxClipRingVerts];
  int32_t const nSubj =
      loadExteriorRingXY(aData, aLen, subj, kMaxClipRingVerts, invalidTypeFlag);
  int32_t const nClip =
      loadExteriorRingXY(bData, bLen, clip, kMaxClipRingVerts, invalidTypeFlag);
  if (nSubj < 3 || nClip < 3) {
    return kEmptyPolygonBlobSize;
  }
  int32_t const nOut = sutherlandHodgman(
      subj, nSubj, clip, nClip, out, kMaxClipRingVerts);
  if (nOut < 3) {
    return kEmptyPolygonBlobSize;
  }
  return veloxPolygonBlobSize(1, nOut + 1); // +1 closing vertex
}

__device__ inline bool clipPolygonsWriteBlob(
    char const* aData,
    cudf::size_type aLen,
    char const* bData,
    cudf::size_type bLen,
    char* out,
    int32_t outSize,
    int32_t* invalidTypeFlag) {
  if (outSize < kEmptyPolygonBlobSize) {
    return false;
  }
  double aMinX, aMinY, aMaxX, aMaxY, bMinX, bMinY, bMaxX, bMaxY;
  if (!readEnvelopeFromBlob(
          aData, aLen, aMinX, aMinY, aMaxX, aMaxY, invalidTypeFlag) ||
      !readEnvelopeFromBlob(
          bData, bLen, bMinX, bMinY, bMaxX, bMaxY, invalidTypeFlag)) {
    return false;
  }
  if (!envelopesIntersect(
          aMinX, aMinY, aMaxX, aMaxY, bMinX, bMinY, bMaxX, bMaxY)) {
    writeEmptyPolygonBlob(out);
    return true;
  }
  double subj[2 * kMaxClipRingVerts];
  double clip[2 * kMaxClipRingVerts];
  double clipped[2 * kMaxClipRingVerts];
  int32_t const nSubj =
      loadExteriorRingXY(aData, aLen, subj, kMaxClipRingVerts, invalidTypeFlag);
  int32_t const nClip =
      loadExteriorRingXY(bData, bLen, clip, kMaxClipRingVerts, invalidTypeFlag);
  if (nSubj < 3 || nClip < 3) {
    writeEmptyPolygonBlob(out);
    return true;
  }
  int32_t const nOut = sutherlandHodgman(
      subj, nSubj, clip, nClip, clipped, kMaxClipRingVerts);
  if (nOut < 3) {
    writeEmptyPolygonBlob(out);
    return true;
  }
  int32_t const need = veloxPolygonBlobSize(1, nOut + 1);
  if (outSize < need) {
    return false;
  }
  writeSingleRingPolygonBlob(out, clipped, nOut);
  return true;
}

/// Inspect WKB and return Velox blob byte size. Returns false → null row.
__device__ inline bool wkbVeloxOutputSize(
    char const* wkb,
    cudf::size_type len,
    int32_t* outSize,
    int32_t* invalidTypeFlag) {
  if (len < 5) {
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
  bool const hasSrid = (typeWord & kWkbSridFlag) != 0;
  uint32_t const geomType = typeWord & 0xFFu;
  cudf::size_type body = 5;
  if (hasSrid) {
    if (len < 9) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    body = 9;
  }

  if (geomType == 1) {
    if (len < body + 16) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    *outSize = kPointBlobSize;
    return true;
  }
  if (geomType == 3) {
    // Polygon: numRings + rings.
    if (len < body + 4) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    uint32_t const numRings = readU32(wkb + body, le);
    if (numRings > 100000u) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    cudf::size_type off = body + 4;
    int32_t numPoints = 0;
    for (uint32_t r = 0; r < numRings; ++r) {
      if (off + 4 > len) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      uint32_t const n = readU32(wkb + off, le);
      off += 4;
      if (n > 10000000u ||
          off + static_cast<cudf::size_type>(n) * 16 > len) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      off += static_cast<cudf::size_type>(n) * 16;
      numPoints += static_cast<int32_t>(n);
    }
    int32_t const numParts = static_cast<int32_t>(numRings);
    *outSize = (numParts == 0) ? kEmptyPolygonBlobSize
                               : veloxPolygonBlobSize(numParts, numPoints);
    return true;
  }
  if (geomType == 6) {
    // MultiPolygon: numPolygons + each polygon (numRings + rings).
    if (len < body + 4) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    uint32_t const numPolygons = readU32(wkb + body, le);
    if (numPolygons > 100000u) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    cudf::size_type off = body + 4;
    int32_t numParts = 0;
    int32_t numPoints = 0;
    for (uint32_t g = 0; g < numPolygons; ++g) {
      if (off + 1 > len) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      // Embedded polygon may repeat endian+type.
      uint8_t const polyOrder = static_cast<uint8_t>(wkb[off]);
      if (polyOrder > 1) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      bool const polyLe = polyOrder == 1;
      if (off + 5 > len) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      uint32_t polyTypeWord = readU32(wkb + off + 1, polyLe);
      uint32_t const polyType = polyTypeWord & 0xFFu;
      cudf::size_type polyBody = off + 5;
      if ((polyTypeWord & kWkbSridFlag) != 0) {
        if (off + 9 > len) {
          markInvalid(invalidTypeFlag);
          return false;
        }
        polyBody = off + 9;
      }
      if (polyType != 3) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      if (polyBody + 4 > len) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      uint32_t const numRings = readU32(wkb + polyBody, polyLe);
      off = polyBody + 4;
      for (uint32_t r = 0; r < numRings; ++r) {
        if (off + 4 > len) {
          markInvalid(invalidTypeFlag);
          return false;
        }
        uint32_t const n = readU32(wkb + off, polyLe);
        off += 4;
        if (n > 10000000u ||
            off + static_cast<cudf::size_type>(n) * 16 > len) {
          markInvalid(invalidTypeFlag);
          return false;
        }
        off += static_cast<cudf::size_type>(n) * 16;
        numPoints += static_cast<int32_t>(n);
        ++numParts;
      }
    }
    *outSize = (numParts == 0) ? kEmptyPolygonBlobSize
                               : veloxPolygonBlobSize(numParts, numPoints);
    return true;
  }
  markInvalid(invalidTypeFlag);
  return false;
}

__device__ inline bool wkbToVeloxBlob(
    char const* wkb,
    cudf::size_type len,
    char* out,
    int32_t outSize,
    int32_t* invalidTypeFlag) {
  if (len < 5) {
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
  bool const hasSrid = (typeWord & kWkbSridFlag) != 0;
  uint32_t const geomType = typeWord & 0xFFu;
  cudf::size_type body = 5;
  if (hasSrid) {
    if (len < 9) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    body = 9;
  }

  if (geomType == 1) {
    if (outSize < kPointBlobSize || len < body + 16) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    double const x = readF64(wkb + body, le);
    double const y = readF64(wkb + body + 8, le);
    out[0] = static_cast<char>(kPointTag);
    writeF64Native(out + 1, x);
    writeF64Native(out + 9, y);
    return true;
  }
  if (geomType == 3) {
    if (len < body + 4) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    uint32_t const numRings = readU32(wkb + body, le);
    int32_t const numParts = static_cast<int32_t>(numRings);
    if (numParts == 0) {
      if (outSize < kEmptyPolygonBlobSize) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      out[0] = static_cast<char>(kPolygonTag);
      writeI32Native(out + 1, kEsriPolygon);
      double const nan = NAN;
      writeF64Native(out + 5, nan);
      writeF64Native(out + 13, nan);
      writeF64Native(out + 21, nan);
      writeF64Native(out + 29, nan);
      writeI32Native(out + 37, 0);
      writeI32Native(out + 41, 0);
      return true;
    }

    // Count points (already validated in size pass, re-check bounds).
    cudf::size_type off = body + 4;
    int32_t numPoints = 0;
    for (int32_t r = 0; r < numParts; ++r) {
      if (off + 4 > len) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      uint32_t const n = readU32(wkb + off, le);
      off += 4 + static_cast<cudf::size_type>(n) * 16;
      numPoints += static_cast<int32_t>(n);
    }
    int32_t const need = veloxPolygonBlobSize(numParts, numPoints);
    if (outSize < need) {
      markInvalid(invalidTypeFlag);
      return false;
    }

    out[0] = static_cast<char>(kPolygonTag);
    writeI32Native(out + 1, kEsriPolygon);
    writeI32Native(out + 37, numParts);
    writeI32Native(out + 41, numPoints);

    char* partPtr = out + kEmptyPolygonBlobSize;
    char* xyPtr = partPtr + 4 * numParts;
    double xmin = INFINITY;
    double ymin = INFINITY;
    double xmax = -INFINITY;
    double ymax = -INFINITY;

    off = body + 4;
    int32_t pointIdx = 0;
    for (int32_t r = 0; r < numParts; ++r) {
      writeI32Native(partPtr + 4 * r, pointIdx);
      uint32_t const n = readU32(wkb + off, le);
      off += 4;
      for (uint32_t i = 0; i < n; ++i) {
        double const x = readF64(wkb + off, le);
        double const y = readF64(wkb + off + 8, le);
        off += 16;
        xmin = fmin(xmin, x);
        ymin = fmin(ymin, y);
        xmax = fmax(xmax, x);
        ymax = fmax(ymax, y);
        writeF64Native(xyPtr + static_cast<std::size_t>(pointIdx) * 16, x);
        writeF64Native(xyPtr + static_cast<std::size_t>(pointIdx) * 16 + 8, y);
        ++pointIdx;
      }
    }
    writeF64Native(out + 5, xmin);
    writeF64Native(out + 13, ymin);
    writeF64Native(out + 21, xmax);
    writeF64Native(out + 29, ymax);
    return true;
  }
  if (geomType == 6) {
    // MultiPolygon → Velox MULTI_POLYGON (tag 5), shells CW / holes CCW.
    if (len < body + 4) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    uint32_t const numPolygons = readU32(wkb + body, le);
    // First pass: count parts/points and validate.
    cudf::size_type off = body + 4;
    int32_t numParts = 0;
    int32_t numPoints = 0;
    for (uint32_t g = 0; g < numPolygons; ++g) {
      if (off + 5 > len) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      uint8_t const polyOrder = static_cast<uint8_t>(wkb[off]);
      if (polyOrder > 1) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      bool const polyLe = polyOrder == 1;
      uint32_t polyTypeWord = readU32(wkb + off + 1, polyLe);
      uint32_t const polyType = polyTypeWord & 0xFFu;
      cudf::size_type polyBody = off + 5;
      if ((polyTypeWord & kWkbSridFlag) != 0) {
        if (off + 9 > len) {
          markInvalid(invalidTypeFlag);
          return false;
        }
        polyBody = off + 9;
      }
      if (polyType != 3 || polyBody + 4 > len) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      uint32_t const numRings = readU32(wkb + polyBody, polyLe);
      off = polyBody + 4;
      for (uint32_t r = 0; r < numRings; ++r) {
        if (off + 4 > len) {
          markInvalid(invalidTypeFlag);
          return false;
        }
        uint32_t const n = readU32(wkb + off, polyLe);
        off += 4 + static_cast<cudf::size_type>(n) * 16;
        numPoints += static_cast<int32_t>(n);
        ++numParts;
      }
    }
    if (numParts == 0) {
      if (outSize < kEmptyPolygonBlobSize) {
        markInvalid(invalidTypeFlag);
        return false;
      }
      out[0] = static_cast<char>(kMultiPolygonTag);
      writeI32Native(out + 1, kEsriPolygon);
      double const nan = NAN;
      writeF64Native(out + 5, nan);
      writeF64Native(out + 13, nan);
      writeF64Native(out + 21, nan);
      writeF64Native(out + 29, nan);
      writeI32Native(out + 37, 0);
      writeI32Native(out + 41, 0);
      return true;
    }
    int32_t const need = veloxPolygonBlobSize(numParts, numPoints);
    if (outSize < need) {
      markInvalid(invalidTypeFlag);
      return false;
    }

    out[0] = static_cast<char>(kMultiPolygonTag);
    writeI32Native(out + 1, kEsriPolygon);
    writeI32Native(out + 37, numParts);
    writeI32Native(out + 41, numPoints);
    char* partPtr = out + kEmptyPolygonBlobSize;
    char* xyPtr = partPtr + 4 * numParts;
    double xmin = INFINITY;
    double ymin = INFINITY;
    double xmax = -INFINITY;
    double ymax = -INFINITY;

    off = body + 4;
    int32_t pointIdx = 0;
    int32_t partIdx = 0;
    for (uint32_t g = 0; g < numPolygons; ++g) {
      uint8_t const polyOrder = static_cast<uint8_t>(wkb[off]);
      bool const polyLe = polyOrder == 1;
      uint32_t polyTypeWord = readU32(wkb + off + 1, polyLe);
      cudf::size_type polyBody = off + 5;
      if ((polyTypeWord & kWkbSridFlag) != 0) {
        polyBody = off + 9;
      }
      uint32_t const numRings = readU32(wkb + polyBody, polyLe);
      off = polyBody + 4;
      for (uint32_t r = 0; r < numRings; ++r) {
        writeI32Native(partPtr + 4 * partIdx, pointIdx);
        uint32_t const n = readU32(wkb + off, polyLe);
        off += 4;
        int32_t const ringStart = pointIdx;
        for (uint32_t i = 0; i < n; ++i) {
          double const x = readF64(wkb + off, polyLe);
          double const y = readF64(wkb + off + 8, polyLe);
          off += 16;
          xmin = fmin(xmin, x);
          ymin = fmin(ymin, y);
          xmax = fmax(xmax, x);
          ymax = fmax(ymax, y);
          writeF64Native(xyPtr + static_cast<std::size_t>(pointIdx) * 16, x);
          writeF64Native(
              xyPtr + static_cast<std::size_t>(pointIdx) * 16 + 8, y);
          ++pointIdx;
        }
        int32_t const ringEnd = pointIdx;
        bool const isShell = (r == 0);
        bool const cw = ringIsClockwiseBytes(xyPtr, ringStart, ringEnd);
        // Shells must be CW, holes CCW (Velox GeometrySerde).
        if ((isShell && !cw) || (!isShell && cw)) {
          reverseRingBytes(xyPtr, ringStart, ringEnd);
        }
        ++partIdx;
      }
    }
    writeF64Native(out + 5, xmin);
    writeF64Native(out + 13, ymin);
    writeF64Native(out + 21, xmax);
    writeF64Native(out + 29, ymax);
    return true;
  }
  markInvalid(invalidTypeFlag);
  return false;
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
  auto offsets = cudf::detail::offsetalator_factory::make_input_iterator(strings.offsets());
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
  auto leftOffsets = cudf::detail::offsetalator_factory::make_input_iterator(leftStr.offsets());
  auto rightOffsets = cudf::detail::offsetalator_factory::make_input_iterator(rightStr.offsets());
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
  // offsets: size+1 entries, each row is exactly kPointBlobSize bytes. Built via
  // make_offsets_child_column so the child promotes to 64-bit offsets when
  // size * kPointBlobSize exceeds 2^31 (past ~126M points).
  auto const sizeItr =
      thrust::make_constant_iterator<cudf::size_type>(kPointBlobSize);
  auto [offsetsCol, totalCharsI64] =
      cudf::strings::detail::make_offsets_child_column(
          sizeItr, sizeItr + size, stream, mr);

  rmm::device_uvector<char> chars(
      static_cast<std::size_t>(totalCharsI64), stream, mr);
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

std::unique_ptr<cudf::column> wkbToVeloxGeometry(
    cudf::column_view const& wkb,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      wkb.type().id() == cudf::type_id::STRING,
      "ST_GeomFromBinary expects VARBINARY/STRING WKB");
  CUDF_EXPECTS(
      wkb.num_children() >= 1 || wkb.size() == 0,
      "ST_GeomFromBinary: STRING column missing children");

  auto const size = wkb.size();
  if (size == 0) {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }

  cudf::strings_column_view strings(wkb);

  auto inChars = strings.chars_begin(stream);
  auto inOffsets = cudf::detail::offsetalator_factory::make_input_iterator(strings.offsets());
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

  // sizes[0..size-1] = blob bytes; sizes[size]=0 for exclusive_scan → offsets.
  rmm::device_uvector<cudf::size_type> sizes(size + 1, stream, mr);
  auto* sizesPtr = sizes.data();
  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size + 1,
      [sizesPtr,
       inChars,
       inOffsets,
       inNull,
       outMask,
       invalidTypeFlag,
       size,
       inNullCount = wkb.null_count()] __device__(cudf::size_type i) {
        if (i == size) {
          sizesPtr[i] = 0;
          return;
        }
        if (inNullCount > 0 && inNull != nullptr &&
            !cudf::bit_is_set(inNull, i)) {
          sizesPtr[i] = 0;
          return;
        }
        int32_t blobSize = 0;
        auto const start = inOffsets[i];
        auto const end = inOffsets[i + 1];
        if (!wkbVeloxOutputSize(
                inChars + start, end - start, &blobSize, invalidTypeFlag)) {
          sizesPtr[i] = 0;
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        sizesPtr[i] = static_cast<cudf::size_type>(blobSize);
      });

  // Build the offsets child, promoting to 64-bit offsets automatically when the
  // total velox-geometry output exceeds 2^31 bytes (e.g. ~1M zone polygons in
  // Q11 at SF100). sizes has size+1 entries; only the first `size` are strings.
  auto [offsetsCol, totalCharsI64] =
      cudf::strings::detail::make_offsets_child_column(
          sizes.begin(), sizes.begin() + size, stream, mr);
  auto outOffsets = cudf::detail::offsetalator_factory::make_input_iterator(
      offsetsCol->view());

  rmm::device_uvector<char> chars(
      static_cast<std::size_t>(totalCharsI64), stream, mr);
  auto* charsPtr = chars.data();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [charsPtr,
       outOffsets,
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
        if (!cudf::bit_is_set(outMask, i)) {
          return;
        }
        auto const outStart = outOffsets[i];
        auto const outEnd = outOffsets[i + 1];
        auto const outLen = outEnd - outStart;
        if (outLen <= 0) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        auto const start = inOffsets[i];
        auto const end = inOffsets[i + 1];
        if (!wkbToVeloxBlob(
                inChars + start,
                end - start,
                charsPtr + outStart,
                static_cast<int32_t>(outLen),
                invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
        }
      });

  auto nullCount = cudf::null_count(
      static_cast<cudf::bitmask_type const*>(nullMaskBuf.data()),
      0,
      size,
      stream);
  return cudf::make_strings_column(
      size,
      std::move(offsetsCol),
      chars.release(),
      nullCount,
      std::move(nullMaskBuf));
}

namespace {
// Copy a single offset value from a device offsets column back to the host,
// honoring its actual width. Columns whose char data exceeds 2^31 bytes store
// 64-bit offsets; reading them as int32 yields garbage byte ranges.
int64_t readOffsetHost(
    cudf::column_view const& offsets,
    cudf::size_type i,
    rmm::cuda_stream_view stream) {
  if (offsets.type().id() == cudf::type_id::INT64) {
    int64_t v = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &v,
        offsets.data<int64_t>() + i,
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    return v;
  }
  int32_t v = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &v,
      offsets.data<int32_t>() + i,
      sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  return static_cast<int64_t>(v);
}
} // namespace

bool geometryColumnLooksLikeWkb(
    cudf::column_view const& geometry,
    rmm::cuda_stream_view stream) {
  if (geometry.size() == 0 || geometry.type().id() != cudf::type_id::STRING) {
    return false;
  }
  cudf::strings_column_view sv(geometry);
  auto const offsetsCol = sv.offsets();
  auto const* chars = sv.chars_begin(stream);
  // Sample up to 8 rows. Do NOT touch geometry.null_mask() on the host —
  // it is device memory (SIGSEGV / "invalid permissions for mapped object").
  constexpr cudf::size_type kSample = 8;
  constexpr cudf::size_type kVeloxPointLen = 17;
  constexpr uint8_t kVeloxPoint = 0;
  constexpr uint8_t kVeloxLineString = 2;
  constexpr uint8_t kVeloxMultiLineString = 3;
  constexpr uint8_t kVeloxPolygon = 4;
  constexpr uint8_t kVeloxMultiPolygon = 5;
  constexpr uint8_t kVeloxEnvelope = 7;
  cudf::size_type checked = 0;
  auto const n = std::min(geometry.size(), kSample);
  for (cudf::size_type i = 0; i < n; ++i) {
    auto const start = readOffsetHost(offsetsCol, i, stream);
    auto const end = readOffsetHost(offsetsCol, i + 1, stream);
    auto const len = end - start;
    if (len < 1) {
      continue; // null/empty
    }
    uint8_t first = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &first,
        chars + start,
        sizeof(uint8_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    ++checked;
    // WKB little-endian always starts with 0x01.
    if (first == 1) {
      return true;
    }
    // WKB big-endian starts with 0x00; Velox POINT is also tag 0 with len 17.
    if (first == 0 && len != kVeloxPointLen) {
      return true;
    }
    if (first == kVeloxPoint || first == kVeloxLineString ||
        first == kVeloxMultiLineString || first == kVeloxPolygon ||
        first == kVeloxMultiPolygon || first == kVeloxEnvelope) {
      continue;
    }
    return true;
  }
  return false;
}

bool geometryColumnLooksLikePoints(
    cudf::column_view const& geometry,
    rmm::cuda_stream_view stream) {
  if (geometry.size() == 0 || geometry.type().id() != cudf::type_id::STRING) {
    return false;
  }
  cudf::strings_column_view sv(geometry);
  auto const offsetsCol = sv.offsets();
  auto const* chars = sv.chars_begin(stream);
  // Do not dereference device null_mask on the host (see looksLikeWkb).
  constexpr cudf::size_type kSample = 8;
  constexpr cudf::size_type kVeloxPointLen = 17;
  constexpr cudf::size_type kWkbPointLen = 21;
  cudf::size_type points = 0;
  cudf::size_type checked = 0;
  auto const n = std::min(geometry.size(), kSample);
  for (cudf::size_type i = 0; i < n; ++i) {
    auto const start = readOffsetHost(offsetsCol, i, stream);
    auto const end = readOffsetHost(offsetsCol, i + 1, stream);
    auto const len = end - start;
    if (len < 1) {
      continue;
    }
    uint8_t first = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &first,
        chars + start,
        sizeof(uint8_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    ++checked;
    // Velox POINT tag 0 length 17, or WKB Point (LE/BE) length 21.
    if ((first == 0 && len == kVeloxPointLen) ||
        (first <= 1 && len == kWkbPointLen)) {
      ++points;
    }
  }
  return checked > 0 && points * 2 >= checked; // majority points
}

std::unique_ptr<cudf::column> ensureVeloxGeometryColumn(
    cudf::column_view const& geometry,
    bool forceWkb,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (forceWkb || geometryColumnLooksLikeWkb(geometry, stream)) {
    return wkbToVeloxGeometry(geometry, invalidTypeFlag, stream, mr);
  }
  return nullptr;
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
  auto offsets = cudf::detail::offsetalator_factory::make_input_iterator(strings.offsets());
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

std::unique_ptr<cudf::column> geometryDistance(
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
  auto outMask =
      static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());

  auto leftChars = leftStr.chars_begin(stream);
  auto rightChars = rightStr.chars_begin(stream);
  auto leftOffsets = cudf::detail::offsetalator_factory::make_input_iterator(leftStr.offsets());
  auto rightOffsets = cudf::detail::offsetalator_factory::make_input_iterator(rightStr.offsets());
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
        auto const ls = leftOffsets[i];
        auto const le = leftOffsets[i + 1];
        auto const rs = rightOffsets[i];
        auto const re = rightOffsets[i + 1];
        char const* leftData = leftChars + ls;
        char const* rightData = rightChars + rs;
        auto const leftLen = le - ls;
        auto const rightLen = re - rs;
        if (leftLen < 1 || rightLen < 1) {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        uint8_t const leftTag = static_cast<uint8_t>(leftData[0]);
        uint8_t const rightTag = static_cast<uint8_t>(rightData[0]);

        // POINT–POINT
        if (leftTag == kPointTag && rightTag == kPointTag) {
          double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
          if (!readPointXY(leftData, leftLen, x1, y1, invalidTypeFlag) ||
              !readPointXY(rightData, rightLen, x2, y2, invalidTypeFlag)) {
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
          return;
        }

        // POINT–POLYGON / ENVELOPE / MULTI_POLYGON (either order)
        bool leftIsPoint = leftTag == kPointTag;
        bool rightIsPoly = rightTag == kPolygonTag ||
            rightTag == kEnvelopeTag || rightTag == kMultiPolygonTag;
        bool rightIsPoint = rightTag == kPointTag;
        bool leftIsPoly = leftTag == kPolygonTag || leftTag == kEnvelopeTag ||
            leftTag == kMultiPolygonTag;

        double px = 0;
        double py = 0;
        char const* polyData = nullptr;
        cudf::size_type polyLen = 0;
        if (leftIsPoint && rightIsPoly) {
          if (!readPointXY(leftData, leftLen, px, py, invalidTypeFlag)) {
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          polyData = rightData;
          polyLen = rightLen;
        } else if (leftIsPoly && rightIsPoint) {
          if (!readPointXY(rightData, rightLen, px, py, invalidTypeFlag)) {
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          polyData = leftData;
          polyLen = leftLen;
        } else {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        if (isEmptyPoint(px, py)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        double dist = 0;
        if (!distPointToPolygonBlob(
                px, py, polyData, polyLen, dist, invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        outPtr[i] = dist;
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

std::unique_ptr<cudf::column> pointToConstantPolygonWithin(
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
      cudf::data_type{cudf::type_id::BOOL8},
      size,
      cudf::mask_state::ALL_VALID,
      stream,
      mr);
  auto* outPtr = out->mutable_view().data<bool>();
  auto outMask =
      static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());

  auto chars = strings.chars_begin(stream);
  auto offsets = cudf::detail::offsetalator_factory::make_input_iterator(strings.offsets());
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
        if (!inShell) {
          outPtr[i] = false;
          return;
        }
        for (int32_t p = 1; p < numParts; ++p) {
          int32_t hs = partEnds[p - 1];
          int32_t he = partEnds[p];
          if (pointInRing(px, py, xy, hs, he)) {
            outPtr[i] = false;
            return;
          }
        }
        outPtr[i] = true;
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

std::unique_ptr<cudf::column> geometryWithinPointsVsPolygon(
    cudf::column_view const& points,
    cudf::column_view const& polygon,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      points.type().id() == cudf::type_id::STRING &&
          polygon.type().id() == cudf::type_id::STRING,
      "geometry inputs must be STRING/VARBINARY");
  CUDF_EXPECTS(polygon.size() == 1, "polygon column must have size 1");

  cudf::strings_column_view pointStr(points);
  cudf::strings_column_view polyStr(polygon);
  auto const size = points.size();
  auto out = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8},
      size,
      cudf::mask_state::ALL_VALID,
      stream,
      mr);
  if (size == 0) {
    return out;
  }
  auto* outPtr = out->mutable_view().data<bool>();
  auto outMask =
      static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());

  auto pointChars = pointStr.chars_begin(stream);
  auto pointOffsets = cudf::detail::offsetalator_factory::make_input_iterator(pointStr.offsets());
  auto pointNull = points.null_mask();
  auto polyChars = polyStr.chars_begin(stream);

  // Capture single polygon blob bounds on the host (dtype-aware: >2GB columns
  // store 64-bit offsets).
  auto const polyStart = readOffsetHost(polyStr.offsets(), 0, stream);
  auto const polyEnd = readOffsetHost(polyStr.offsets(), 1, stream);
  char const* polyData = polyChars + polyStart;
  auto const polyLen = polyEnd - polyStart;
  if (polygon.null_count() > 0 || polyLen < 1) {
    thrust::for_each_n(
        rmm::exec_policy(stream),
        thrust::counting_iterator<cudf::size_type>(0),
        size,
        [outMask] __device__(cudf::size_type i) {
          cudf::clear_bit_unsafe(outMask, i);
        });
    out->set_null_count(size);
    return out;
  }

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [pointChars,
       pointOffsets,
       pointNull,
       outPtr,
       outMask,
       invalidTypeFlag,
       polyData,
       polyLen,
       pointNullCount = points.null_count()] __device__(cudf::size_type i) {
        if (pointNullCount > 0 && pointNull != nullptr &&
            !cudf::bit_is_set(pointNull, i)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        auto const ps = pointOffsets[i];
        auto const pe = pointOffsets[i + 1];
        char const* pointData = pointChars + ps;
        auto const pointLen = pe - ps;
        if (pointLen < 1) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        if (static_cast<uint8_t>(pointData[0]) != kPointTag) {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        double px = 0;
        double py = 0;
        if (!readPointXY(pointData, pointLen, px, py, invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        if (isEmptyPoint(px, py)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        bool inside = false;
        if (!pointInPolygonBlob(
                px, py, polyData, polyLen, inside, invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        outPtr[i] = inside;
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

std::unique_ptr<cudf::column> geometryWithinIndexed(
    cudf::column_view const& points,
    cudf::column_view const& polygons,
    cudf::column_view const& pointIndices,
    cudf::column_view const& polyIndices,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      points.type().id() == cudf::type_id::STRING &&
          polygons.type().id() == cudf::type_id::STRING,
      "geometry inputs must be STRING/VARBINARY");
  CUDF_EXPECTS(
      pointIndices.type().id() == cudf::type_to_id<cudf::size_type>() &&
          polyIndices.type().id() == cudf::type_to_id<cudf::size_type>(),
      "indices must be size_type");
  CUDF_EXPECTS(
      pointIndices.size() == polyIndices.size(), "index size mismatch");

  auto const size = pointIndices.size();
  auto out = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8},
      size,
      cudf::mask_state::ALL_VALID,
      stream,
      mr);
  if (size == 0) {
    return out;
  }
  auto* outPtr = out->mutable_view().data<bool>();
  auto outMask =
      static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());

  cudf::strings_column_view pointStr(points);
  cudf::strings_column_view polyStr(polygons);
  auto pointChars = pointStr.chars_begin(stream);
  auto polyChars = polyStr.chars_begin(stream);
  auto pointOffsets = cudf::detail::offsetalator_factory::make_input_iterator(pointStr.offsets());
  auto polyOffsets = cudf::detail::offsetalator_factory::make_input_iterator(polyStr.offsets());
  auto pointNull = points.null_mask();
  auto polyNull = polygons.null_mask();
  auto const* pointIdx = pointIndices.data<cudf::size_type>();
  auto const* polyIdx = polyIndices.data<cudf::size_type>();
  auto const nPoints = points.size();
  auto const nPolys = polygons.size();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [pointChars,
       polyChars,
       pointOffsets,
       polyOffsets,
       pointNull,
       polyNull,
       pointIdx,
       polyIdx,
       outPtr,
       outMask,
       invalidTypeFlag,
       nPoints,
       nPolys,
       pointNullCount = points.null_count(),
       polyNullCount = polygons.null_count()] __device__(cudf::size_type i) {
        auto const pi = pointIdx[i];
        auto const bi = polyIdx[i];
        if (pi < 0 || pi >= nPoints || bi < 0 || bi >= nPolys) {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        if ((pointNullCount > 0 && pointNull != nullptr &&
             !cudf::bit_is_set(pointNull, pi)) ||
            (polyNullCount > 0 && polyNull != nullptr &&
             !cudf::bit_is_set(polyNull, bi))) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        auto const ps = pointOffsets[pi];
        auto const pe = pointOffsets[pi + 1];
        auto const bs = polyOffsets[bi];
        auto const be = polyOffsets[bi + 1];
        char const* pointData = pointChars + ps;
        char const* polyData = polyChars + bs;
        auto const pointLen = pe - ps;
        auto const polyLen = be - bs;
        if (pointLen < 1 || polyLen < 1) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        uint8_t const pointTag = static_cast<uint8_t>(pointData[0]);
        uint8_t const polyTag = static_cast<uint8_t>(polyData[0]);
        // Accept either argument order for robustness.
        char const* pData = nullptr;
        cudf::size_type pLen = 0;
        char const* gData = nullptr;
        cudf::size_type gLen = 0;
        if (pointTag == kPointTag &&
            (polyTag == kPolygonTag || polyTag == kEnvelopeTag ||
             polyTag == kMultiPolygonTag)) {
          pData = pointData;
          pLen = pointLen;
          gData = polyData;
          gLen = polyLen;
        } else if (
            polyTag == kPointTag &&
            (pointTag == kPolygonTag || pointTag == kEnvelopeTag ||
             pointTag == kMultiPolygonTag)) {
          pData = polyData;
          pLen = polyLen;
          gData = pointData;
          gLen = pointLen;
        } else {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        double px = 0;
        double py = 0;
        if (!readPointXY(pData, pLen, px, py, invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        if (isEmptyPoint(px, py)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        bool inside = false;
        if (!pointInPolygonBlob(
                px, py, gData, gLen, inside, invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        outPtr[i] = inside;
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

std::unique_ptr<cudf::column> geometryWithinIndexedXY(
    cudf::column_view const& pointX,
    cudf::column_view const& pointY,
    cudf::column_view const& polygons,
    cudf::column_view const& pointIndices,
    cudf::column_view const& polyIndices,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      pointX.type().id() == cudf::type_id::FLOAT64 &&
          pointY.type().id() == cudf::type_id::FLOAT64,
      "point XY must be FLOAT64");
  CUDF_EXPECTS(
      polygons.type().id() == cudf::type_id::STRING,
      "polygons must be STRING/VARBINARY");
  CUDF_EXPECTS(
      pointX.size() == pointY.size(), "point XY size mismatch");
  CUDF_EXPECTS(
      pointIndices.type().id() == cudf::type_to_id<cudf::size_type>() &&
          polyIndices.type().id() == cudf::type_to_id<cudf::size_type>(),
      "indices must be size_type");
  CUDF_EXPECTS(
      pointIndices.size() == polyIndices.size(), "index size mismatch");

  auto const size = pointIndices.size();
  auto out = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8},
      size,
      cudf::mask_state::ALL_VALID,
      stream,
      mr);
  if (size == 0) {
    return out;
  }
  auto* outPtr = out->mutable_view().data<bool>();
  auto outMask =
      static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());

  cudf::strings_column_view polyStr(polygons);
  auto polyChars = polyStr.chars_begin(stream);
  auto polyOffsets =
      cudf::detail::offsetalator_factory::make_input_iterator(polyStr.offsets());
  auto polyNull = polygons.null_mask();
  auto pointNull = pointX.null_mask();
  auto const* xPtr = pointX.data<double>();
  auto const* yPtr = pointY.data<double>();
  auto const* pointIdx = pointIndices.data<cudf::size_type>();
  auto const* polyIdx = polyIndices.data<cudf::size_type>();
  auto const nPoints = pointX.size();
  auto const nPolys = polygons.size();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [polyChars,
       polyOffsets,
       polyNull,
       pointNull,
       xPtr,
       yPtr,
       pointIdx,
       polyIdx,
       outPtr,
       outMask,
       invalidTypeFlag,
       nPoints,
       nPolys,
       pointNullCount = pointX.null_count(),
       polyNullCount = polygons.null_count()] __device__(cudf::size_type i) {
        auto const pi = pointIdx[i];
        auto const bi = polyIdx[i];
        if (pi < 0 || pi >= nPoints || bi < 0 || bi >= nPolys) {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        if ((pointNullCount > 0 && pointNull != nullptr &&
             !cudf::bit_is_set(pointNull, pi)) ||
            (polyNullCount > 0 && polyNull != nullptr &&
             !cudf::bit_is_set(polyNull, bi))) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        double const px = xPtr[pi];
        double const py = yPtr[pi];
        if (isEmptyPoint(px, py) || isnan(px) || isnan(py)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        auto const bs = polyOffsets[bi];
        auto const be = polyOffsets[bi + 1];
        char const* polyData = polyChars + bs;
        auto const polyLen = be - bs;
        if (polyLen < 1) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        uint8_t const polyTag = static_cast<uint8_t>(polyData[0]);
        if (polyTag != kPolygonTag && polyTag != kEnvelopeTag &&
            polyTag != kMultiPolygonTag) {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        bool inside = false;
        if (!pointInPolygonBlob(
                px, py, polyData, polyLen, inside, invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        outPtr[i] = inside;
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

std::unique_ptr<cudf::column> geometryWithin(
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
      cudf::data_type{cudf::type_id::BOOL8},
      size,
      cudf::mask_state::ALL_VALID,
      stream,
      mr);
  auto* outPtr = out->mutable_view().data<bool>();
  auto outMask =
      static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());

  auto leftChars = leftStr.chars_begin(stream);
  auto rightChars = rightStr.chars_begin(stream);
  auto leftOffsets = cudf::detail::offsetalator_factory::make_input_iterator(leftStr.offsets());
  auto rightOffsets = cudf::detail::offsetalator_factory::make_input_iterator(rightStr.offsets());
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
        auto const ls = leftOffsets[i];
        auto const le = leftOffsets[i + 1];
        auto const rs = rightOffsets[i];
        auto const re = rightOffsets[i + 1];
        char const* leftData = leftChars + ls;
        char const* rightData = rightChars + rs;
        auto const leftLen = le - ls;
        auto const rightLen = re - rs;
        if (leftLen < 1 || rightLen < 1) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        uint8_t const leftTag = static_cast<uint8_t>(leftData[0]);
        uint8_t const rightTag = static_cast<uint8_t>(rightData[0]);
        // Accept POINT–POLYGON either order (SpatialJoin may flip sides).
        bool const leftIsPoint = leftTag == kPointTag;
        bool const rightIsPoly = rightTag == kPolygonTag ||
            rightTag == kEnvelopeTag || rightTag == kMultiPolygonTag;
        bool const rightIsPoint = rightTag == kPointTag;
        bool const leftIsPoly = leftTag == kPolygonTag ||
            leftTag == kEnvelopeTag || leftTag == kMultiPolygonTag;

        double px = 0;
        double py = 0;
        char const* polyData = nullptr;
        cudf::size_type polyLen = 0;
        if (leftIsPoint && rightIsPoly) {
          if (!readPointXY(leftData, leftLen, px, py, invalidTypeFlag)) {
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          polyData = rightData;
          polyLen = rightLen;
        } else if (leftIsPoly && rightIsPoint) {
          if (!readPointXY(rightData, rightLen, px, py, invalidTypeFlag)) {
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          polyData = leftData;
          polyLen = leftLen;
        } else {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        if (isEmptyPoint(px, py)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        bool inside = false;
        if (!pointInPolygonBlob(
                px, py, polyData, polyLen, inside, invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        outPtr[i] = inside;
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

std::unique_ptr<cudf::column> geometryIntersects(
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
      cudf::data_type{cudf::type_id::BOOL8},
      size,
      cudf::mask_state::ALL_VALID,
      stream,
      mr);
  auto* outPtr = out->mutable_view().data<bool>();
  auto outMask =
      static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());

  auto leftChars = leftStr.chars_begin(stream);
  auto rightChars = rightStr.chars_begin(stream);
  auto leftOffsets = cudf::detail::offsetalator_factory::make_input_iterator(leftStr.offsets());
  auto rightOffsets = cudf::detail::offsetalator_factory::make_input_iterator(rightStr.offsets());
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
        auto const ls = leftOffsets[i];
        auto const le = leftOffsets[i + 1];
        auto const rs = rightOffsets[i];
        auto const re = rightOffsets[i + 1];
        char const* leftData = leftChars + ls;
        char const* rightData = rightChars + rs;
        auto const leftLen = le - ls;
        auto const rightLen = re - rs;
        if (leftLen < 1 || rightLen < 1) {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        uint8_t const leftTag = static_cast<uint8_t>(leftData[0]);
        uint8_t const rightTag = static_cast<uint8_t>(rightData[0]);
        bool const leftIsPoint = leftTag == kPointTag;
        bool const rightIsPoint = rightTag == kPointTag;
        bool const leftIsPoly =
            leftTag == kPolygonTag || leftTag == kEnvelopeTag ||
            leftTag == kMultiPolygonTag;
        bool const rightIsPoly =
            rightTag == kPolygonTag || rightTag == kEnvelopeTag ||
            rightTag == kMultiPolygonTag;

        if ((leftIsPoint && rightIsPoly) || (leftIsPoly && rightIsPoint)) {
          double px = 0;
          double py = 0;
          char const* polyData = nullptr;
          cudf::size_type polyLen = 0;
          if (leftIsPoint) {
            if (!readPointXY(leftData, leftLen, px, py, invalidTypeFlag)) {
              cudf::clear_bit_unsafe(outMask, i);
              return;
            }
            polyData = rightData;
            polyLen = rightLen;
          } else {
            if (!readPointXY(rightData, rightLen, px, py, invalidTypeFlag)) {
              cudf::clear_bit_unsafe(outMask, i);
              return;
            }
            polyData = leftData;
            polyLen = leftLen;
          }
          if (isEmptyPoint(px, py)) {
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          bool inside = false;
          if (!pointInPolygonBlob(
                  px, py, polyData, polyLen, inside, invalidTypeFlag)) {
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          outPtr[i] = inside;
          return;
        }

        if (leftIsPoly && rightIsPoly) {
          bool hit = false;
          if (!polygonsIntersectBlob(
                  leftData,
                  leftLen,
                  rightData,
                  rightLen,
                  hit,
                  invalidTypeFlag)) {
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          outPtr[i] = hit;
          return;
        }

        markInvalid(invalidTypeFlag);
        cudf::clear_bit_unsafe(outMask, i);
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

std::unique_ptr<cudf::column> geometryIntersectsConstantPolygon(
    cudf::column_view const& geometry,
    DevicePolygonView polygon,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      geometry.type().id() == cudf::type_id::STRING,
      "geometry must be STRING/VARBINARY");
  CUDF_EXPECTS(geometry.num_children() >= 1 || geometry.size() == 0,
               "geometry strings column missing children");
  CUDF_EXPECTS(polygon.numParts > 0 && polygon.numPoints > 0, "empty polygon");

  auto const size = geometry.size();
  auto out = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8},
      size,
      cudf::mask_state::ALL_VALID,
      stream,
      mr);
  if (size == 0) {
    out->set_null_count(0);
    return out;
  }

  auto* outPtr = out->mutable_view().data<bool>();
  auto outMask =
      static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());
  cudf::strings_column_view strings(geometry);
  auto chars = strings.chars_begin(stream);
  auto offsets = cudf::detail::offsetalator_factory::make_input_iterator(strings.offsets());
  auto inNull = geometry.null_mask();
  auto const* xy = polygon.xy;
  auto const* partEnds = polygon.partEnds;
  auto const numParts = polygon.numParts;
  auto const numPoints = polygon.numPoints;

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
       numPoints,
       nullCount = geometry.null_count()] __device__(cudf::size_type i) {
        if (nullCount > 0 && inNull != nullptr &&
            !cudf::bit_is_set(inNull, i)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        auto const start = offsets[i];
        auto const end = offsets[i + 1];
        char const* data = chars + start;
        auto const len = end - start;
        if (len < 1) {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        uint8_t const tag = static_cast<uint8_t>(data[0]);

        // Constant envelope.
        double cMinX = INFINITY, cMinY = INFINITY, cMaxX = -INFINITY,
               cMaxY = -INFINITY;
        for (int32_t p = 0; p < numPoints; ++p) {
          double const x = xy[2 * p];
          double const y = xy[2 * p + 1];
          cMinX = fmin(cMinX, x);
          cMinY = fmin(cMinY, y);
          cMaxX = fmax(cMaxX, x);
          cMaxY = fmax(cMaxY, y);
        }

        auto pointInConst = [&](double px, double py) {
          int32_t shellStart = 0;
          int32_t shellEnd = partEnds[0];
          if (!pointInRing(px, py, xy, shellStart, shellEnd)) {
            return false;
          }
          for (int32_t p = 1; p < numParts; ++p) {
            if (pointInRing(px, py, xy, partEnds[p - 1], partEnds[p])) {
              return false;
            }
          }
          return true;
        };

        if (tag == kPointTag) {
          double px = 0, py = 0;
          if (!readPointXY(data, len, px, py, invalidTypeFlag) ||
              isEmptyPoint(px, py)) {
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          outPtr[i] = pointInConst(px, py);
          return;
        }

        if (tag == kPolygonTag || tag == kMultiPolygonTag ||
            tag == kEnvelopeTag) {
          double eMinX = 0, eMinY = 0, eMaxX = 0, eMaxY = 0;
          if (!readEnvelopeFromBlob(
                  data, len, eMinX, eMinY, eMaxX, eMaxY, invalidTypeFlag)) {
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          if (!envelopesIntersect(
                  eMinX, eMinY, eMaxX, eMaxY, cMinX, cMinY, cMaxX, cMaxY)) {
            outPtr[i] = false;
            return;
          }
          // Vertex of row geometry inside constant, or constant corners inside
          // row geometry.
          char envScratch[80];
          char const* rowXy = nullptr;
          int32_t rowParts = 0, rowPoints = 0;
          bool rowEnv = false;
          if (!parsePolygonRings(
                  data,
                  len,
                  rowXy,
                  envScratch,
                  rowParts,
                  rowPoints,
                  rowEnv,
                  invalidTypeFlag)) {
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          for (int32_t p = 0; p < rowPoints; ++p) {
            if (pointInConst(xyBytesX(rowXy, p), xyBytesY(rowXy, p))) {
              outPtr[i] = true;
              return;
            }
          }
          for (int32_t p = 0; p < numPoints; ++p) {
            bool inside = false;
            if (!pointInPolygonBlob(
                    xy[2 * p],
                    xy[2 * p + 1],
                    data,
                    len,
                    inside,
                    invalidTypeFlag)) {
              cudf::clear_bit_unsafe(outMask, i);
              return;
            }
            if (inside) {
              outPtr[i] = true;
              return;
            }
          }
          outPtr[i] = false;
          return;
        }

        markInvalid(invalidTypeFlag);
        cudf::clear_bit_unsafe(outMask, i);
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

std::unique_ptr<cudf::column> makeLineStringFromPointList(
    cudf::column_view const& pointLists,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      pointLists.type().id() == cudf::type_id::LIST,
      "ST_LineString expects array(geometry)");

  cudf::lists_column_view lists(pointLists);
  auto const size = pointLists.size();
  auto points = lists.get_sliced_child(stream);
  CUDF_EXPECTS(
      points.type().id() == cudf::type_id::STRING,
      "ST_LineString array elements must be geometry/STRING");

  cudf::strings_column_view pointStrings(points);
  auto pointChars = pointStrings.chars_begin(stream);
  auto pointOffsets = cudf::detail::offsetalator_factory::make_input_iterator(pointStrings.offsets());
  auto pointNull = points.null_mask();
  auto listOffsets = lists.offsets().begin<cudf::size_type>();
  auto listNull = pointLists.null_mask();

  auto [nullMaskBuf, nullCountHint] = [&]() {
    if (pointLists.null_mask() != nullptr) {
      auto buf = cudf::copy_bitmask(pointLists, stream, mr);
      return std::make_pair(std::move(buf), pointLists.null_count());
    }
    return std::make_pair(
        cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr),
        0);
  }();
  (void)nullCountHint;
  auto* outMask = static_cast<cudf::bitmask_type*>(nullMaskBuf.data());

  rmm::device_uvector<cudf::size_type> sizes(size + 1, stream, mr);
  auto* sizesPtr = sizes.data();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size + 1,
      [sizesPtr,
       listOffsets,
       listNull,
       pointOffsets,
       pointChars,
       pointNull,
       outMask,
       invalidTypeFlag,
       size,
       listNullCount = pointLists.null_count(),
       pointNullCount = points.null_count()] __device__(cudf::size_type i) {
        if (i == size) {
          sizesPtr[i] = 0;
          return;
        }
        if (listNullCount > 0 && listNull != nullptr &&
            !cudf::bit_is_set(listNull, i)) {
          sizesPtr[i] = 0;
          return;
        }
        auto const begin = listOffsets[i];
        auto const end = listOffsets[i + 1];
        auto const n = end - begin;
        if (n < 2) {
          sizesPtr[i] = kEmptyLineStringBlobSize;
          return;
        }
        // Validate points; size the blob.
        for (cudf::size_type j = begin; j < end; ++j) {
          if (pointNullCount > 0 && pointNull != nullptr &&
              !cudf::bit_is_set(pointNull, j)) {
            markInvalid(invalidTypeFlag);
            sizesPtr[i] = 0;
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          double x = 0;
          double y = 0;
          auto const ps = pointOffsets[j];
          auto const pe = pointOffsets[j + 1];
          if (!readPointXY(pointChars + ps, pe - ps, x, y, invalidTypeFlag)) {
            sizesPtr[i] = 0;
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          if (isEmptyPoint(x, y)) {
            markInvalid(invalidTypeFlag);
            sizesPtr[i] = 0;
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          // Consecutive duplicates allowed (Sedona ST_MakeLine / Q7 parity).
        }
        sizesPtr[i] = static_cast<cudf::size_type>(
            veloxLineStringBlobSize(static_cast<int32_t>(n)));
      });

  auto offsetsCol = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      size + 1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* offsets = offsetsCol->mutable_view().data<cudf::size_type>();
  thrust::exclusive_scan(
      rmm::exec_policy(stream), sizes.begin(), sizes.end(), offsets);

  cudf::size_type totalChars = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &totalChars,
      offsets + size,
      sizeof(cudf::size_type),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  rmm::device_uvector<char> chars(
      static_cast<std::size_t>(totalChars), stream, mr);
  auto* charsPtr = chars.data();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [charsPtr,
       offsets,
       listOffsets,
       listNull,
       pointOffsets,
       pointChars,
       pointNull,
       outMask,
       invalidTypeFlag,
       listNullCount = pointLists.null_count(),
       pointNullCount = points.null_count()] __device__(cudf::size_type i) {
        if (listNullCount > 0 && listNull != nullptr &&
            !cudf::bit_is_set(listNull, i)) {
          return;
        }
        if (!cudf::bit_is_set(outMask, i)) {
          return;
        }
        auto const outStart = offsets[i];
        auto const outEnd = offsets[i + 1];
        auto const outLen = outEnd - outStart;
        if (outLen <= 0) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        char* out = charsPtr + outStart;
        auto const begin = listOffsets[i];
        auto const end = listOffsets[i + 1];
        auto const n = end - begin;

        if (n < 2) {
          // Empty LINESTRING.
          out[0] = static_cast<char>(kLineStringTag);
          writeI32Native(out + 1, kEsriPolyline);
          double const nan = NAN;
          writeF64Native(out + 5, nan);
          writeF64Native(out + 13, nan);
          writeF64Native(out + 21, nan);
          writeF64Native(out + 29, nan);
          writeI32Native(out + 37, 0);
          writeI32Native(out + 41, 0);
          return;
        }

        out[0] = static_cast<char>(kLineStringTag);
        writeI32Native(out + 1, kEsriPolyline);
        writeI32Native(out + 37, 1); // numParts
        writeI32Native(out + 41, static_cast<int32_t>(n));
        writeI32Native(out + 45, 0); // partStarts[0]
        char* xyPtr = out + 49;

        double xmin = INFINITY;
        double ymin = INFINITY;
        double xmax = -INFINITY;
        double ymax = -INFINITY;
        int32_t pointIdx = 0;
        for (cudf::size_type j = begin; j < end; ++j) {
          double x = 0;
          double y = 0;
          auto const ps = pointOffsets[j];
          auto const pe = pointOffsets[j + 1];
          if (!readPointXY(pointChars + ps, pe - ps, x, y, invalidTypeFlag)) {
            cudf::clear_bit_unsafe(outMask, i);
            return;
          }
          xmin = fmin(xmin, x);
          ymin = fmin(ymin, y);
          xmax = fmax(xmax, x);
          ymax = fmax(ymax, y);
          writeF64Native(xyPtr + static_cast<std::size_t>(pointIdx) * 16, x);
          writeF64Native(
              xyPtr + static_cast<std::size_t>(pointIdx) * 16 + 8, y);
          ++pointIdx;
        }
        writeF64Native(out + 5, xmin);
        writeF64Native(out + 13, ymin);
        writeF64Native(out + 21, xmax);
        writeF64Native(out + 29, ymax);
      });

  auto nullCount = cudf::null_count(
      static_cast<cudf::bitmask_type const*>(nullMaskBuf.data()),
      0,
      size,
      stream);
  return cudf::make_strings_column(
      size,
      std::move(offsetsCol),
      chars.release(),
      nullCount,
      std::move(nullMaskBuf));
}

std::unique_ptr<cudf::column> lineStringLength(
    cudf::column_view const& geometry,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      geometry.type().id() == cudf::type_id::STRING,
      "ST_Length expects geometry/STRING");

  cudf::strings_column_view strings(geometry);
  auto const size = geometry.size();
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
  auto offsets = cudf::detail::offsetalator_factory::make_input_iterator(strings.offsets());
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
       invalidTypeFlag,
       nullCount = geometry.null_count()] __device__(cudf::size_type i) {
        if (nullCount > 0 && inNull != nullptr &&
            !cudf::bit_is_set(inNull, i)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        auto const start = offsets[i];
        auto const end = offsets[i + 1];
        auto const len = end - start;
        char const* data = chars + start;
        if (len < kEmptyLineStringBlobSize) {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        uint8_t const tag = static_cast<uint8_t>(data[0]);
        if (tag != kLineStringTag && tag != kMultiLineStringTag) {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        int32_t const numParts = readI32Native(data + 37);
        int32_t const numPoints = readI32Native(data + 41);
        if (numParts <= 0 || numPoints <= 1) {
          outPtr[i] = 0.0;
          return;
        }
        std::size_t const partsBytes =
            static_cast<std::size_t>(numParts) * sizeof(int32_t);
        std::size_t const xyBytes =
            static_cast<std::size_t>(numPoints) * 16;
        if (static_cast<std::size_t>(len) <
            static_cast<std::size_t>(kEmptyLineStringBlobSize) + partsBytes +
                xyBytes) {
          markInvalid(invalidTypeFlag);
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        char const* partPtr = data + kEmptyLineStringBlobSize;
        char const* xy = partPtr + partsBytes;

        double total = 0.0;
        for (int32_t p = 0; p < numParts; ++p) {
          int32_t const ps = readI32Native(
              partPtr + static_cast<std::size_t>(p) * 4);
          int32_t const pe = (p + 1 < numParts)
              ? readI32Native(
                    partPtr + static_cast<std::size_t>(p + 1) * 4)
              : numPoints;
          for (int32_t k = ps; k + 1 < pe; ++k) {
            double const x0 = xyBytesX(xy, k);
            double const y0 = xyBytesY(xy, k);
            double const x1 = xyBytesX(xy, k + 1);
            double const y1 = xyBytesY(xy, k + 1);
            double const dx = x1 - x0;
            double const dy = y1 - y0;
            total += sqrt(dx * dx + dy * dy);
          }
        }
        outPtr[i] = total;
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

std::unique_ptr<cudf::column> geometryArea(
    cudf::column_view const& geometry,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      geometry.type().id() == cudf::type_id::STRING,
      "ST_Area expects geometry/STRING");

  auto const size = geometry.size();
  auto out = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::FLOAT64},
      size,
      cudf::mask_state::ALL_VALID,
      stream,
      mr);
  if (size == 0) {
    return out;
  }
  // All-null STRING placeholders from upstream may lack offsets/chars children.
  if (geometry.null_count() == size) {
    auto outMask =
        static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());
    if (geometry.null_mask() != nullptr) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          outMask,
          geometry.null_mask(),
          cudf::bitmask_allocation_size_bytes(size),
          cudaMemcpyDeviceToDevice,
          stream.value()));
    } else {
      thrust::for_each_n(
          rmm::exec_policy(stream),
          thrust::counting_iterator<cudf::size_type>(0),
          size,
          [outMask] __device__(cudf::size_type i) {
            cudf::clear_bit_unsafe(outMask, i);
          });
    }
    out->set_null_count(size);
    return out;
  }

  cudf::strings_column_view strings(geometry);
  auto* outPtr = out->mutable_view().data<double>();
  auto outMask =
      static_cast<cudf::bitmask_type*>(out->mutable_view().null_mask());

  auto chars = strings.chars_begin(stream);
  auto offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(strings.offsets());
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
       invalidTypeFlag,
       nullCount = geometry.null_count()] __device__(cudf::size_type i) {
        if (nullCount > 0 && inNull != nullptr &&
            !cudf::bit_is_set(inNull, i)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        auto const start = offsets[i];
        auto const end = offsets[i + 1];
        auto const len = static_cast<cudf::size_type>(end - start);
        char const* data = chars + start;
        if (len < 1) {
          outPtr[i] = 0.0;
          return;
        }
        uint8_t const tag = static_cast<uint8_t>(data[0]);
        // GEOS getArea(): non-areal geometries contribute 0.
        if (tag == kPointTag || tag == kMultiPointTag ||
            tag == kLineStringTag || tag == kMultiLineStringTag) {
          outPtr[i] = 0.0;
          return;
        }

        char envScratch[80];
        char const* xy = nullptr;
        int32_t numParts = 0;
        int32_t numPoints = 0;
        bool envelope = false;

        if (tag == kEnvelopeTag) {
          if (len < 1 + 32) {
            markInvalid(invalidTypeFlag);
            outPtr[i] = 0.0;
            return;
          }
          double const xmin = readF64Native(data + 1);
          double const ymin = readF64Native(data + 9);
          double const xmax = readF64Native(data + 17);
          double const ymax = readF64Native(data + 25);
          double coords[10] = {
              xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin};
#pragma unroll
          for (int j = 0; j < 10; ++j) {
            writeF64Native(envScratch + j * 8, coords[j]);
          }
          xy = envScratch;
          numParts = 1;
          numPoints = 5;
          envelope = true;
        } else if (tag == kPolygonTag || tag == kMultiPolygonTag) {
          if (len < kEmptyPolygonBlobSize) {
            markInvalid(invalidTypeFlag);
            outPtr[i] = 0.0;
            return;
          }
          numParts = readI32Native(data + 37);
          numPoints = readI32Native(data + 41);
          if (numParts <= 0 || numPoints <= 0) {
            outPtr[i] = 0.0;
            return;
          }
          std::size_t const partsBytes =
              static_cast<std::size_t>(numParts) * sizeof(int32_t);
          std::size_t const xyBytesNeeded =
              static_cast<std::size_t>(numPoints) * 16;
          if (static_cast<std::size_t>(len) <
              static_cast<std::size_t>(kEmptyPolygonBlobSize) + partsBytes +
                  xyBytesNeeded) {
            markInvalid(invalidTypeFlag);
            outPtr[i] = 0.0;
            return;
          }
          xy = data + kEmptyPolygonBlobSize + partsBytes;
        } else {
          markInvalid(invalidTypeFlag);
          outPtr[i] = 0.0;
          return;
        }

        // Signed shoelace (CCW-positive). Esri shells are CW and holes CCW, so
        // sum(signed) = −shell + holes; abs recovers GEOS shell−holes area.
        double signedArea = 0.0;
        for (int32_t p = 0; p < numParts; ++p) {
          int32_t ps = 0;
          int32_t pe = 0;
          if (envelope) {
            ps = 0;
            pe = numPoints;
          } else {
            ps = readI32Native(
                data + kEmptyPolygonBlobSize +
                static_cast<std::size_t>(p) * 4);
            pe = (p + 1 < numParts)
                ? readI32Native(
                      data + kEmptyPolygonBlobSize +
                      static_cast<std::size_t>(p + 1) * 4)
                : numPoints;
          }
          if (pe - ps < 3) {
            continue;
          }
          double ringSum = 0.0;
          for (int32_t k = ps; k + 1 < pe; ++k) {
            double const x0 = xyBytesX(xy, k);
            double const y0 = xyBytesY(xy, k);
            double const x1 = xyBytesX(xy, k + 1);
            double const y1 = xyBytesY(xy, k + 1);
            ringSum += x0 * y1 - x1 * y0;
          }
          double const xLast = xyBytesX(xy, pe - 1);
          double const yLast = xyBytesY(xy, pe - 1);
          double const xFirst = xyBytesX(xy, ps);
          double const yFirst = xyBytesY(xy, ps);
          if (xLast != xFirst || yLast != yFirst) {
            ringSum += xLast * yFirst - xFirst * yLast;
          }
          signedArea += 0.5 * ringSum;
        }
        outPtr[i] = fabs(signedArea);
      });

  out->set_null_count(
      cudf::null_count(out->view().null_mask(), 0, size, stream));
  return out;
}

std::unique_ptr<cudf::column> geometryIntersection(
    cudf::column_view const& left,
    cudf::column_view const& right,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      left.type().id() == cudf::type_id::STRING &&
          right.type().id() == cudf::type_id::STRING,
      "ST_Intersection expects geometry/STRING columns");
  CUDF_EXPECTS(left.size() == right.size(), "ST_Intersection size mismatch");

  cudf::strings_column_view leftStr(left);
  cudf::strings_column_view rightStr(right);
  auto const size = left.size();

  auto nullMaskBuf =
      cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
  auto* outMask = static_cast<cudf::bitmask_type*>(nullMaskBuf.data());

  rmm::device_uvector<cudf::size_type> sizes(
      static_cast<std::size_t>(size) + 1, stream, mr);
  auto* sizesPtr = sizes.data();

  auto leftChars = leftStr.chars_begin(stream);
  auto rightChars = rightStr.chars_begin(stream);
  auto leftOff =
      cudf::detail::offsetalator_factory::make_input_iterator(leftStr.offsets());
  auto rightOff =
      cudf::detail::offsetalator_factory::make_input_iterator(rightStr.offsets());
  auto leftNull = left.null_mask();
  auto rightNull = right.null_mask();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size + 1,
      [sizesPtr,
       outMask,
       leftChars,
       rightChars,
       leftOff,
       rightOff,
       leftNull,
       rightNull,
       invalidTypeFlag,
       size,
       leftNullCount = left.null_count(),
       rightNullCount = right.null_count()] __device__(cudf::size_type i) {
        if (i == size) {
          sizesPtr[i] = 0;
          return;
        }
        if ((leftNullCount > 0 && leftNull != nullptr &&
             !cudf::bit_is_set(leftNull, i)) ||
            (rightNullCount > 0 && rightNull != nullptr &&
             !cudf::bit_is_set(rightNull, i))) {
          sizesPtr[i] = 0;
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        auto const ls = leftOff[i];
        auto const le = leftOff[i + 1];
        auto const rs = rightOff[i];
        auto const re = rightOff[i + 1];
        int32_t const sz = clipPolygonsBlobSize(
            leftChars + ls,
            static_cast<cudf::size_type>(le - ls),
            rightChars + rs,
            static_cast<cudf::size_type>(re - rs),
            invalidTypeFlag);
        if (sz <= 0) {
          sizesPtr[i] = 0;
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        sizesPtr[i] = static_cast<cudf::size_type>(sz);
      });

  auto offsetsCol = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      size + 1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* offsets = offsetsCol->mutable_view().data<cudf::size_type>();
  thrust::exclusive_scan(
      rmm::exec_policy(stream), sizes.begin(), sizes.end(), offsets);

  cudf::size_type totalChars = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &totalChars,
      offsets + size,
      sizeof(cudf::size_type),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  rmm::device_uvector<char> chars(
      static_cast<std::size_t>(totalChars), stream, mr);
  auto* charsPtr = chars.data();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [charsPtr,
       offsets,
       outMask,
       leftChars,
       rightChars,
       leftOff,
       rightOff,
       invalidTypeFlag] __device__(cudf::size_type i) {
        if (!cudf::bit_is_set(outMask, i)) {
          return;
        }
        auto const outStart = offsets[i];
        auto const outEnd = offsets[i + 1];
        auto const outLen = outEnd - outStart;
        if (outLen <= 0) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        auto const ls = leftOff[i];
        auto const le = leftOff[i + 1];
        auto const rs = rightOff[i];
        auto const re = rightOff[i + 1];
        if (!clipPolygonsWriteBlob(
                leftChars + ls,
                static_cast<cudf::size_type>(le - ls),
                rightChars + rs,
                static_cast<cudf::size_type>(re - rs),
                charsPtr + outStart,
                outLen,
                invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
        }
      });

  auto nullCount = cudf::null_count(outMask, 0, size, stream);
  return cudf::make_strings_column(
      size,
      std::move(offsetsCol),
      chars.release(),
      nullCount,
      std::move(nullMaskBuf));
}

namespace {

constexpr int32_t kEsriMultiPoint = 8;
constexpr int32_t kMultiPointHeaderSize = 41; // tag+esri+envelope+numPoints
constexpr int32_t kMaxHullPoints = 1024;

__device__ inline int32_t veloxMultiPointBlobSize(int32_t numPoints) {
  return kMultiPointHeaderSize + 16 * numPoints;
}

__device__ inline void writeMultiPointBlob(
    char* out,
    double const* xy,
    int32_t n) {
  out[0] = static_cast<char>(kMultiPointTag);
  writeI32Native(out + 1, kEsriMultiPoint);
  double xmin = INFINITY, ymin = INFINITY, xmax = -INFINITY, ymax = -INFINITY;
  char* xyPtr = out + kMultiPointHeaderSize;
  for (int32_t i = 0; i < n; ++i) {
    double const x = xy[2 * i];
    double const y = xy[2 * i + 1];
    writeF64Native(xyPtr + static_cast<std::size_t>(i) * 16, x);
    writeF64Native(xyPtr + static_cast<std::size_t>(i) * 16 + 8, y);
    xmin = fmin(xmin, x);
    ymin = fmin(ymin, y);
    xmax = fmax(xmax, x);
    ymax = fmax(ymax, y);
  }
  writeF64Native(out + 5, xmin);
  writeF64Native(out + 13, ymin);
  writeF64Native(out + 21, xmax);
  writeF64Native(out + 29, ymax);
  writeI32Native(out + 37, n);
}

__device__ inline void writePointBlob(char* out, double x, double y) {
  out[0] = static_cast<char>(kPointTag);
  writeF64Native(out + 1, x);
  writeF64Native(out + 9, y);
}

__device__ inline void writeLineStringTwoPointBlob(
    char* out,
    double x0,
    double y0,
    double x1,
    double y1) {
  out[0] = static_cast<char>(kLineStringTag);
  writeI32Native(out + 1, kEsriPolyline);
  writeF64Native(out + 5, fmin(x0, x1));
  writeF64Native(out + 13, fmin(y0, y1));
  writeF64Native(out + 21, fmax(x0, x1));
  writeF64Native(out + 29, fmax(y0, y1));
  writeI32Native(out + 37, 1); // numParts
  writeI32Native(out + 41, 2); // numPoints
  writeI32Native(out + kEmptyLineStringBlobSize, 0); // partStarts[0]
  char* xyPtr = out + kEmptyLineStringBlobSize + 4;
  writeF64Native(xyPtr, x0);
  writeF64Native(xyPtr + 8, y0);
  writeF64Native(xyPtr + 16, x1);
  writeF64Native(xyPtr + 24, y1);
}

/// Load POINT blobs from a list slice into xy[2*n]. Returns count or -1 on
/// unsupported / overflow (marks invalid).
template <typename OffsetIter>
__device__ inline int32_t loadPointsFromListSlice(
    char const* pointChars,
    OffsetIter pointOffsets,
    cudf::bitmask_type const* pointNull,
    int32_t pointNullCount,
    cudf::size_type begin,
    cudf::size_type end,
    double* xy,
    int32_t maxPoints,
    int32_t* invalidTypeFlag) {
  int32_t n = 0;
  for (cudf::size_type j = begin; j < end; ++j) {
    if (pointNullCount > 0 && pointNull != nullptr &&
        !cudf::bit_is_set(pointNull, j)) {
      continue; // geometry_union ignores nulls
    }
    auto const ps = pointOffsets[j];
    auto const pe = pointOffsets[j + 1];
    auto const len = static_cast<cudf::size_type>(pe - ps);
    if (len < 1) {
      markInvalid(invalidTypeFlag);
      return -1;
    }
    uint8_t const tag = static_cast<uint8_t>(pointChars[ps]);
    if (tag != kPointTag) {
      markInvalid(invalidTypeFlag);
      return -1;
    }
    double x = 0, y = 0;
    if (!readPointXY(pointChars + ps, len, x, y, invalidTypeFlag)) {
      return -1;
    }
    if (isEmptyPoint(x, y)) {
      continue; // skip empties; all-empty handled by caller
    }
    if (n >= maxPoints) {
      markInvalid(invalidTypeFlag);
      return -1;
    }
    xy[2 * n] = x;
    xy[2 * n + 1] = y;
    ++n;
  }
  return n;
}

/// Load coordinates from POINT or MULTI_POINT blob into xy. Returns n or -1.
__device__ inline int32_t loadPointsFromGeometryBlob(
    char const* data,
    cudf::size_type len,
    double* xy,
    int32_t maxPoints,
    int32_t* invalidTypeFlag) {
  if (len < 1) {
    markInvalid(invalidTypeFlag);
    return -1;
  }
  uint8_t const tag = static_cast<uint8_t>(data[0]);
  if (tag == kPointTag) {
    double x = 0, y = 0;
    if (!readPointXY(data, len, x, y, invalidTypeFlag)) {
      return -1;
    }
    if (isEmptyPoint(x, y)) {
      return 0;
    }
    if (maxPoints < 1) {
      markInvalid(invalidTypeFlag);
      return -1;
    }
    xy[0] = x;
    xy[1] = y;
    return 1;
  }
  if (tag == kMultiPointTag) {
    if (len < kMultiPointHeaderSize) {
      markInvalid(invalidTypeFlag);
      return -1;
    }
    int32_t const numPoints = readI32Native(data + 37);
    if (numPoints < 0 ||
        len < kMultiPointHeaderSize + 16 * numPoints) {
      markInvalid(invalidTypeFlag);
      return -1;
    }
    if (numPoints > maxPoints) {
      markInvalid(invalidTypeFlag);
      return -1;
    }
    char const* xyPtr = data + kMultiPointHeaderSize;
    int32_t n = 0;
    for (int32_t i = 0; i < numPoints; ++i) {
      double const x = readF64Native(xyPtr + static_cast<std::size_t>(i) * 16);
      double const y =
          readF64Native(xyPtr + static_cast<std::size_t>(i) * 16 + 8);
      if (isEmptyPoint(x, y) || isnan(x) || isnan(y)) {
        continue;
      }
      xy[2 * n] = x;
      xy[2 * n + 1] = y;
      ++n;
    }
    return n;
  }
  markInvalid(invalidTypeFlag);
  return -1;
}

__device__ inline double cross2d(
    double ox,
    double oy,
    double ax,
    double ay,
    double bx,
    double by) {
  return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}

/// Sort (x,y) lexicographically in-place (insertion sort; n ≤ kMaxHullPoints).
__device__ inline void sortPointsXY(double* xy, int32_t n) {
  for (int32_t i = 1; i < n; ++i) {
    double const x = xy[2 * i];
    double const y = xy[2 * i + 1];
    int32_t j = i - 1;
    while (j >= 0 &&
           (xy[2 * j] > x || (xy[2 * j] == x && xy[2 * j + 1] > y))) {
      xy[2 * (j + 1)] = xy[2 * j];
      xy[2 * (j + 1) + 1] = xy[2 * j + 1];
      --j;
    }
    xy[2 * (j + 1)] = x;
    xy[2 * (j + 1) + 1] = y;
  }
}

/// Dedupe sorted points; returns new count.
__device__ inline int32_t uniquePointsXY(double* xy, int32_t n) {
  if (n <= 0) {
    return 0;
  }
  int32_t w = 1;
  for (int32_t i = 1; i < n; ++i) {
    if (xy[2 * i] != xy[2 * (w - 1)] ||
        xy[2 * i + 1] != xy[2 * (w - 1) + 1]) {
      xy[2 * w] = xy[2 * i];
      xy[2 * w + 1] = xy[2 * i + 1];
      ++w;
    }
  }
  return w;
}

/// Andrew's monotone chain. Writes CCW hull vertices (no close) into hullXY.
/// Returns hull vertex count (0 / 1 / 2 / ≥3).
__device__ inline int32_t monotoneChainHull(
    double* xy,
    int32_t n,
    double* hullXY) {
  if (n <= 0) {
    return 0;
  }
  sortPointsXY(xy, n);
  n = uniquePointsXY(xy, n);
  if (n <= 2) {
    for (int32_t i = 0; i < n; ++i) {
      hullXY[2 * i] = xy[2 * i];
      hullXY[2 * i + 1] = xy[2 * i + 1];
    }
    return n;
  }
  int32_t k = 0;
  // Lower hull
  for (int32_t i = 0; i < n; ++i) {
    while (k >= 2 &&
           cross2d(
               hullXY[2 * (k - 2)],
               hullXY[2 * (k - 2) + 1],
               hullXY[2 * (k - 1)],
               hullXY[2 * (k - 1) + 1],
               xy[2 * i],
               xy[2 * i + 1]) <= 0.0) {
      --k;
    }
    hullXY[2 * k] = xy[2 * i];
    hullXY[2 * k + 1] = xy[2 * i + 1];
    ++k;
  }
  // Upper hull
  int32_t const lower = k + 1;
  for (int32_t i = n - 2; i >= 0; --i) {
    while (k >= lower &&
           cross2d(
               hullXY[2 * (k - 2)],
               hullXY[2 * (k - 2) + 1],
               hullXY[2 * (k - 1)],
               hullXY[2 * (k - 1) + 1],
               xy[2 * i],
               xy[2 * i + 1]) <= 0.0) {
      --k;
    }
    hullXY[2 * k] = xy[2 * i];
    hullXY[2 * k + 1] = xy[2 * i + 1];
    ++k;
  }
  // Last point equals first; drop it.
  return k - 1;
}

__device__ inline int32_t convexHullBlobSizeFromPoints(int32_t nUniqueOrHull) {
  if (nUniqueOrHull <= 0) {
    return kEmptyPolygonBlobSize;
  }
  if (nUniqueOrHull == 1) {
    return kPointBlobSize;
  }
  if (nUniqueOrHull == 2) {
    return veloxLineStringBlobSize(2);
  }
  // Closed ring: hullVerts + 1
  return veloxPolygonBlobSize(1, nUniqueOrHull + 1);
}

__device__ inline void writeConvexHullBlob(
    char* out,
    double* xy,
    int32_t n,
    double* scratchHull) {
  int32_t const h = monotoneChainHull(xy, n, scratchHull);
  if (h <= 0) {
    writeEmptyPolygonBlob(out);
    return;
  }
  if (h == 1) {
    writePointBlob(out, scratchHull[0], scratchHull[1]);
    return;
  }
  if (h == 2) {
    writeLineStringTwoPointBlob(
        out,
        scratchHull[0],
        scratchHull[1],
        scratchHull[2],
        scratchHull[3]);
    return;
  }
  writeSingleRingPolygonBlob(out, scratchHull, h);
}

} // namespace

std::unique_ptr<cudf::column> geometryUnionFromList(
    cudf::column_view const& geometryLists,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      geometryLists.type().id() == cudf::type_id::LIST,
      "geometry_union expects array(geometry)");

  cudf::lists_column_view lists(geometryLists);
  auto const size = geometryLists.size();
  auto points = lists.get_sliced_child(stream);
  CUDF_EXPECTS(
      points.type().id() == cudf::type_id::STRING,
      "geometry_union array elements must be geometry/STRING");

  // Empty list child (every ARRAY_AGG empty): no STRING children to view.
  if (size == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }
  if (points.size() == 0) {
    // Every list is empty → Presto geometry_union all-empty → empty POLYGON.
    rmm::device_uvector<cudf::size_type> sizes(
        static_cast<std::size_t>(size) + 1, stream, mr);
    thrust::fill(
        rmm::exec_policy(stream),
        sizes.begin(),
        sizes.begin() + size,
        static_cast<cudf::size_type>(kEmptyPolygonBlobSize));
    {
      cudf::size_type zero = 0;
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          sizes.data() + size,
          &zero,
          sizeof(zero),
          cudaMemcpyHostToDevice,
          stream.value()));
    }
    auto offsetsCol = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        size + 1,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    auto* offsets = offsetsCol->mutable_view().data<cudf::size_type>();
    thrust::exclusive_scan(
        rmm::exec_policy(stream), sizes.begin(), sizes.end(), offsets);
    cudf::size_type totalChars = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &totalChars,
        offsets + size,
        sizeof(cudf::size_type),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    rmm::device_uvector<char> chars(
        static_cast<std::size_t>(totalChars), stream, mr);
    auto* charsPtr = chars.data();
    thrust::for_each_n(
        rmm::exec_policy(stream),
        thrust::counting_iterator<cudf::size_type>(0),
        size,
        [charsPtr, offsets] __device__(cudf::size_type i) {
          writeEmptyPolygonBlob(charsPtr + offsets[i]);
        });
    auto nullMaskBuf = [&]() {
      if (geometryLists.null_mask() != nullptr) {
        return cudf::copy_bitmask(geometryLists, stream, mr);
      }
      return cudf::create_null_mask(
          size, cudf::mask_state::ALL_VALID, stream, mr);
    }();
    auto nullCount = geometryLists.null_count();
    return cudf::make_strings_column(
        size,
        std::move(offsetsCol),
        chars.release(),
        nullCount,
        std::move(nullMaskBuf));
  }

  cudf::strings_column_view pointStrings(points);
  auto pointChars = pointStrings.chars_begin(stream);
  auto pointOffsets =
      cudf::detail::offsetalator_factory::make_input_iterator(
          pointStrings.offsets());
  auto pointNull = points.null_mask();
  auto listOffsets = lists.offsets().begin<cudf::size_type>();
  auto listNull = geometryLists.null_mask();

  auto nullMaskBuf = [&]() {
    if (geometryLists.null_mask() != nullptr) {
      return cudf::copy_bitmask(geometryLists, stream, mr);
    }
    return cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
  }();
  auto* outMask = static_cast<cudf::bitmask_type*>(nullMaskBuf.data());

  rmm::device_uvector<cudf::size_type> sizes(
      static_cast<std::size_t>(size) + 1, stream, mr);
  auto* sizesPtr = sizes.data();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size + 1,
      [sizesPtr,
       listOffsets,
       listNull,
       pointOffsets,
       pointChars,
       pointNull,
       outMask,
       invalidTypeFlag,
       size,
       listNullCount = geometryLists.null_count(),
       pointNullCount = points.null_count()] __device__(cudf::size_type i) {
        if (i == size) {
          sizesPtr[i] = 0;
          return;
        }
        if (listNullCount > 0 && listNull != nullptr &&
            !cudf::bit_is_set(listNull, i)) {
          sizesPtr[i] = 0;
          return;
        }
        double xy[2 * kMaxHullPoints];
        int32_t const n = loadPointsFromListSlice(
            pointChars,
            pointOffsets,
            pointNull,
            pointNullCount,
            listOffsets[i],
            listOffsets[i + 1],
            xy,
            kMaxHullPoints,
            invalidTypeFlag);
        if (n < 0) {
          sizesPtr[i] = 0;
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        if (n == 0) {
          // Presto geometry_union all-empty → empty POLYGON.
          sizesPtr[i] = kEmptyPolygonBlobSize;
          return;
        }
        if (n == 1) {
          sizesPtr[i] = kPointBlobSize;
          return;
        }
        sizesPtr[i] = static_cast<cudf::size_type>(veloxMultiPointBlobSize(n));
      });

  auto offsetsCol = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      size + 1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* offsets = offsetsCol->mutable_view().data<cudf::size_type>();
  thrust::exclusive_scan(
      rmm::exec_policy(stream), sizes.begin(), sizes.end(), offsets);

  cudf::size_type totalChars = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &totalChars,
      offsets + size,
      sizeof(cudf::size_type),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  rmm::device_uvector<char> chars(
      static_cast<std::size_t>(totalChars), stream, mr);
  auto* charsPtr = chars.data();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [charsPtr,
       offsets,
       listOffsets,
       listNull,
       pointOffsets,
       pointChars,
       pointNull,
       outMask,
       invalidTypeFlag,
       listNullCount = geometryLists.null_count(),
       pointNullCount = points.null_count()] __device__(cudf::size_type i) {
        if (listNullCount > 0 && listNull != nullptr &&
            !cudf::bit_is_set(listNull, i)) {
          return;
        }
        if (!cudf::bit_is_set(outMask, i)) {
          return;
        }
        auto const outStart = offsets[i];
        auto const outLen = offsets[i + 1] - outStart;
        if (outLen <= 0) {
          return;
        }
        double xy[2 * kMaxHullPoints];
        int32_t const n = loadPointsFromListSlice(
            pointChars,
            pointOffsets,
            pointNull,
            pointNullCount,
            listOffsets[i],
            listOffsets[i + 1],
            xy,
            kMaxHullPoints,
            invalidTypeFlag);
        char* out = charsPtr + outStart;
        if (n <= 0) {
          writeEmptyPolygonBlob(out);
          return;
        }
        if (n == 1) {
          writePointBlob(out, xy[0], xy[1]);
          return;
        }
        writeMultiPointBlob(out, xy, n);
      });

  auto nullCount = cudf::null_count(outMask, 0, size, stream);
  return cudf::make_strings_column(
      size,
      std::move(offsetsCol),
      chars.release(),
      nullCount,
      std::move(nullMaskBuf));
}

std::unique_ptr<cudf::column> geometryConvexHull(
    cudf::column_view const& geometry,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      geometry.type().id() == cudf::type_id::STRING,
      "ST_ConvexHull expects geometry/STRING column");

  auto const size = geometry.size();
  if (size == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }

  cudf::strings_column_view str(geometry);
  auto charsBegin = str.chars_begin(stream);
  auto offsetsIt =
      cudf::detail::offsetalator_factory::make_input_iterator(str.offsets());
  auto inNull = geometry.null_mask();

  auto nullMaskBuf = [&]() {
    if (geometry.null_mask() != nullptr) {
      return cudf::copy_bitmask(geometry, stream, mr);
    }
    return cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
  }();
  auto* outMask = static_cast<cudf::bitmask_type*>(nullMaskBuf.data());

  rmm::device_uvector<cudf::size_type> sizes(
      static_cast<std::size_t>(size) + 1, stream, mr);
  auto* sizesPtr = sizes.data();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size + 1,
      [sizesPtr,
       charsBegin,
       offsetsIt,
       inNull,
       outMask,
       invalidTypeFlag,
       size,
       nullCount = geometry.null_count()] __device__(cudf::size_type i) {
        if (i == size) {
          sizesPtr[i] = 0;
          return;
        }
        if (nullCount > 0 && inNull != nullptr &&
            !cudf::bit_is_set(inNull, i)) {
          sizesPtr[i] = 0;
          return;
        }
        auto const s = offsetsIt[i];
        auto const e = offsetsIt[i + 1];
        double xy[2 * kMaxHullPoints];
        double hull[2 * kMaxHullPoints];
        int32_t n = loadPointsFromGeometryBlob(
            charsBegin + s,
            static_cast<cudf::size_type>(e - s),
            xy,
            kMaxHullPoints,
            invalidTypeFlag);
        if (n < 0) {
          sizesPtr[i] = 0;
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        // Size after hull (need unique count path): run hull for size.
        int32_t const h = monotoneChainHull(xy, n, hull);
        sizesPtr[i] = static_cast<cudf::size_type>(
            convexHullBlobSizeFromPoints(h));
      });

  auto offsetsCol = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      size + 1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* offsets = offsetsCol->mutable_view().data<cudf::size_type>();
  thrust::exclusive_scan(
      rmm::exec_policy(stream), sizes.begin(), sizes.end(), offsets);

  cudf::size_type totalChars = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &totalChars,
      offsets + size,
      sizeof(cudf::size_type),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  rmm::device_uvector<char> chars(
      static_cast<std::size_t>(totalChars), stream, mr);
  auto* charsPtr = chars.data();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [charsPtr,
       offsets,
       charsBegin,
       offsetsIt,
       inNull,
       outMask,
       invalidTypeFlag,
       nullCount = geometry.null_count()] __device__(cudf::size_type i) {
        if (nullCount > 0 && inNull != nullptr &&
            !cudf::bit_is_set(inNull, i)) {
          return;
        }
        if (!cudf::bit_is_set(outMask, i)) {
          return;
        }
        auto const outStart = offsets[i];
        auto const outLen = offsets[i + 1] - outStart;
        if (outLen <= 0) {
          return;
        }
        auto const s = offsetsIt[i];
        auto const e = offsetsIt[i + 1];
        double xy[2 * kMaxHullPoints];
        double hull[2 * kMaxHullPoints];
        int32_t n = loadPointsFromGeometryBlob(
            charsBegin + s,
            static_cast<cudf::size_type>(e - s),
            xy,
            kMaxHullPoints,
            invalidTypeFlag);
        if (n < 0) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        writeConvexHullBlob(charsPtr + outStart, xy, n, hull);
      });

  auto nullCount = cudf::null_count(outMask, 0, size, stream);
  return cudf::make_strings_column(
      size,
      std::move(offsetsCol),
      chars.release(),
      nullCount,
      std::move(nullMaskBuf));
}

namespace {

__device__ inline bool readEnvelopeFromBlob(
    char const* data,
    cudf::size_type len,
    double& minX,
    double& minY,
    double& maxX,
    double& maxY,
    int32_t* invalidTypeFlag) {
  if (len < 1) {
    markInvalid(invalidTypeFlag);
    return false;
  }
  uint8_t const tag = static_cast<uint8_t>(data[0]);
  if (tag == kPointTag) {
    double x = 0;
    double y = 0;
    if (!readPointXY(data, len, x, y, invalidTypeFlag)) {
      return false;
    }
    if (isEmptyPoint(x, y)) {
      return false;
    }
    minX = maxX = x;
    minY = maxY = y;
    return true;
  }
  if (tag == kEnvelopeTag) {
    if (len < 1 + 32) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    minX = readF64Native(data + 1);
    minY = readF64Native(data + 9);
    maxX = readF64Native(data + 17);
    maxY = readF64Native(data + 25);
  } else if (
      tag == kPolygonTag || tag == kMultiPolygonTag || tag == kLineStringTag ||
      tag == kMultiLineStringTag) {
    // tag(1) + esri(4) + xmin,ymin,xmax,ymax
    if (len < 5 + 32) {
      markInvalid(invalidTypeFlag);
      return false;
    }
    minX = readF64Native(data + 5);
    minY = readF64Native(data + 13);
    maxX = readF64Native(data + 21);
    maxY = readF64Native(data + 29);
  } else {
    markInvalid(invalidTypeFlag);
    return false;
  }
  if (isnan(minX) || isnan(minY) || isnan(maxX) || isnan(maxY)) {
    return false;
  }
  return true;
}

__device__ inline bool envelopesIntersect(
    double aMinX,
    double aMinY,
    double aMaxX,
    double aMaxY,
    double bMinX,
    double bMinY,
    double bMaxX,
    double bMaxY) {
  return (aMaxX >= bMinX) && (aMinX <= bMaxX) && (aMaxY >= bMinY) &&
      (aMinY <= bMaxY);
}

} // namespace

GeometryEnvelopes extractGeometryEnvelopes(
    cudf::column_view const& geometry,
    cudf::column_view const& expandBy,
    double constantExpandBy,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      geometry.type().id() == cudf::type_id::STRING,
      "geometry input must be STRING/VARBINARY");
  bool const perRowExpand = expandBy.size() == geometry.size();
  if (perRowExpand) {
    CUDF_EXPECTS(
        expandBy.type().id() == cudf::type_id::FLOAT64,
        "expandBy must be FLOAT64");
  }

  cudf::strings_column_view geomStr(geometry);
  auto const size = geometry.size();
  auto makeCol = [&]() {
    return cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::FLOAT64},
        size,
        cudf::mask_state::ALL_VALID,
        stream,
        mr);
  };
  GeometryEnvelopes out{
      makeCol(), makeCol(), makeCol(), makeCol()};
  auto* minX = out.minX->mutable_view().data<double>();
  auto* minY = out.minY->mutable_view().data<double>();
  auto* maxX = out.maxX->mutable_view().data<double>();
  auto* maxY = out.maxY->mutable_view().data<double>();
  auto outMask =
      static_cast<cudf::bitmask_type*>(out.minX->mutable_view().null_mask());
  // Share one null mask across all four columns: copy after fill.
  auto chars = geomStr.chars_begin(stream);
  auto offsets = cudf::detail::offsetalator_factory::make_input_iterator(geomStr.offsets());
  auto geomNull = geometry.null_mask();
  auto expandPtr =
      perRowExpand ? expandBy.data<double>() : static_cast<double const*>(nullptr);
  auto expandNull = perRowExpand ? expandBy.null_mask() : nullptr;

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      size,
      [chars,
       offsets,
       geomNull,
       expandPtr,
       expandNull,
       constantExpandBy,
       perRowExpand,
       minX,
       minY,
       maxX,
       maxY,
       outMask,
       invalidTypeFlag] __device__(cudf::size_type i) {
        if (geomNull && !cudf::bit_is_set(geomNull, i)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        if (perRowExpand && expandNull && !cudf::bit_is_set(expandNull, i)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        auto const start = offsets[i];
        auto const end = offsets[i + 1];
        double eMinX = 0, eMinY = 0, eMaxX = 0, eMaxY = 0;
        if (!readEnvelopeFromBlob(
                chars + start,
                end - start,
                eMinX,
                eMinY,
                eMaxX,
                eMaxY,
                invalidTypeFlag)) {
          cudf::clear_bit_unsafe(outMask, i);
          return;
        }
        double radius = constantExpandBy;
        if (perRowExpand) {
          radius = expandPtr[i];
        }
        radius = fmax(radius, 0.0);
        minX[i] = eMinX - radius;
        minY[i] = eMinY - radius;
        maxX[i] = eMaxX + radius;
        maxY[i] = eMaxY + radius;
      });

  auto nullCount =
      cudf::null_count(out.minX->view().null_mask(), 0, size, stream);
  out.minX->set_null_count(nullCount);
  // Propagate the same null mask to the other three columns.
  auto maskBytes = cudf::bitmask_allocation_size_bytes(size);
  for (auto* col : {out.minY.get(), out.maxX.get(), out.maxY.get()}) {
    auto dst = static_cast<cudf::bitmask_type*>(col->mutable_view().null_mask());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        dst,
        out.minX->view().null_mask(),
        maskBytes,
        cudaMemcpyDeviceToDevice,
        stream.value()));
    col->set_null_count(nullCount);
  }
  return out;
}

GeometryPartEnvelopes extractGeometryPartEnvelopes(
    cudf::column_view const& geometry,
    cudf::column_view const& expandBy,
    double constantExpandBy,
    int32_t maxPartsPerRow,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      geometry.type().id() == cudf::type_id::STRING,
      "geometry input must be STRING/VARBINARY [selective-split-v2]");
  bool const perRowExpand = expandBy.size() == geometry.size();
  if (perRowExpand) {
    CUDF_EXPECTS(
        expandBy.type().id() == cudf::type_id::FLOAT64,
        "expandBy must be FLOAT64");
  }
  if (maxPartsPerRow < 1) {
    maxPartsPerRow = 1;
  }

  cudf::strings_column_view geomStr(geometry);
  auto const n = geometry.size();
  auto chars = geomStr.chars_begin(stream);
  auto offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(geomStr.offsets());
  auto geomNull = geometry.null_mask();
  auto expandPtr = perRowExpand ? expandBy.data<double>()
                                : static_cast<double const*>(nullptr);
  auto expandNull = perRowExpand ? expandBy.null_mask() : nullptr;

  // rowPartOffset[r] = number of index parts before row r; length n+1.
  auto rowPartOffset = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64},
      n + 1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* offPtr = rowPartOffset->mutable_view().data<int64_t>();

  // Split threshold: a geometry is "outsized" when its bbox is much wider than
  // the column's average. Deriving it from the data keeps this unit-agnostic
  // (degrees vs. projected metres) instead of hard-coding a degree cutoff.
  // Columns of ordinary same-sized polygons then split nothing, which is the
  // desired no-op.
  constexpr double kSplitSpanMultiplier = 32.0;
  double minSplitSpan = std::numeric_limits<double>::infinity();
  if (n > 0) {
    auto spanIt = thrust::make_transform_iterator(
        thrust::counting_iterator<cudf::size_type>(0),
        cuda::proclaim_return_type<double>(
            [chars, offsets, geomNull] __device__(cudf::size_type i) -> double {
              if (geomNull && !cudf::bit_is_set(geomNull, i)) {
                return 0.0;
              }
              auto const s = offsets[i];
              auto const e = offsets[i + 1];
              return blobBboxSpan(
                  chars + s, static_cast<cudf::size_type>(e - s));
            }));
    double const spanSum = thrust::reduce(
        rmm::exec_policy(stream), spanIt, spanIt + n, 0.0, thrust::plus<double>());
    double const meanSpan = spanSum / static_cast<double>(n);
    if (meanSpan > 0.0) {
      minSplitSpan = kSplitSpanMultiplier * meanSpan;
    }
  }

  // Pass 1: parts-per-row into offPtr[0..n), then exclusive scan over [0..n+1).
  // exclusive_scan output[k] = sum(input[0..k-1]); input[n] is never read for a
  // valid output, so offPtr[n] need not be initialised.
  //
  // Retried once with splitting disabled if the split still inflates the index
  // past a small multiple of the row count (many similarly outsized
  // geometries): a bloated index costs more in query-side 64-bit scratch, which
  // is sized by the index rather than the batch, than tighter boxes win back.
  int64_t const kMaxIndexParts =
      std::max<int64_t>(4 * static_cast<int64_t>(n), 1'000'000);
  int64_t total = 0;
  for (int attempt = 0; attempt < 2; ++attempt) {
    double const splitSpan = minSplitSpan;
    thrust::transform(
        rmm::exec_policy(stream),
        thrust::counting_iterator<cudf::size_type>(0),
        thrust::counting_iterator<cudf::size_type>(n),
        offPtr,
        cuda::proclaim_return_type<int64_t>(
            [chars, offsets, geomNull, maxPartsPerRow, splitSpan] __device__(
                cudf::size_type i) -> int64_t {
              if (geomNull && !cudf::bit_is_set(geomNull, i)) {
                return 0;
              }
              auto const s = offsets[i];
              auto const e = offsets[i + 1];
              return static_cast<int64_t>(blobIndexPartCount(
                  chars + s,
                  static_cast<cudf::size_type>(e - s),
                  maxPartsPerRow,
                  splitSpan));
            }));
    thrust::exclusive_scan(
        rmm::exec_policy(stream), offPtr, offPtr + n + 1, offPtr, int64_t{0});
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &total,
        offPtr + n,
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    if (total <= kMaxIndexParts) {
      break;
    }
    minSplitSpan = std::numeric_limits<double>::infinity();
  }
  LOG_FIRST_N(WARNING, 10) << "[geo:parts] rows=" << n << " total=" << total
                           << " cap=" << kMaxIndexParts
                           << " splitSpan=" << minSplitSpan;

  auto makeCol = [&](int64_t sz) {
    return cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::FLOAT64},
        static_cast<cudf::size_type>(sz),
        cudf::mask_state::ALL_VALID,
        stream,
        mr);
  };
  GeometryPartEnvelopes out;
  out.envelopes = GeometryEnvelopes{
      makeCol(total), makeCol(total), makeCol(total), makeCol(total)};
  out.partToRow = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      static_cast<cudf::size_type>(total),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  out.rowPartOffset = std::move(rowPartOffset);
  if (total == 0) {
    return out;
  }

  auto* minX = out.envelopes.minX->mutable_view().data<double>();
  auto* minY = out.envelopes.minY->mutable_view().data<double>();
  auto* maxX = out.envelopes.maxX->mutable_view().data<double>();
  auto* maxY = out.envelopes.maxY->mutable_view().data<double>();
  auto outMask = static_cast<cudf::bitmask_type*>(
      out.envelopes.minX->mutable_view().null_mask());
  auto* partToRow = out.partToRow->mutable_view().data<cudf::size_type>();
  auto const* offRead = out.rowPartOffset->view().data<int64_t>();

  // Pass 2: one thread per index part. Binary-search the row owning the part,
  // then compute either its ring bbox or the whole-geometry bbox.
  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<int64_t>(0),
      total,
      [chars,
       offsets,
       offRead,
       n,
       maxPartsPerRow,
       minSplitSpan,
       expandPtr,
       expandNull,
       perRowExpand,
       constantExpandBy,
       minX,
       minY,
       maxX,
       maxY,
       outMask,
       partToRow,
       invalidTypeFlag] __device__(int64_t k) {
        // upper_bound - 1 over offRead[0..n] gives the owning row.
        int64_t lo = 0;
        int64_t hi = n; // search in [0, n]
        while (lo < hi) {
          int64_t const mid = (lo + hi + 1) >> 1;
          if (offRead[mid] <= k) {
            lo = mid;
          } else {
            hi = mid - 1;
          }
        }
        auto const row = static_cast<cudf::size_type>(lo);
        partToRow[static_cast<cudf::size_type>(k)] = row;
        auto const localPart =
            static_cast<int32_t>(k - offRead[row]);

        auto const s = offsets[row];
        auto const e = offsets[row + 1];
        char const* data = chars + s;
        auto const len = static_cast<cudf::size_type>(e - s);
        double radius = perRowExpand ? expandPtr[row] : constantExpandBy;
        if (perRowExpand && expandNull &&
            !cudf::bit_is_set(expandNull, row)) {
          radius = 0.0;
        }
        radius = fmax(radius, 0.0);

        double bMinX = 0, bMinY = 0, bMaxX = 0, bMaxY = 0;
        int32_t numParts = 0, numPoints = 0;
        bool ok = false;
        if (blobUsesPerRing(
                data, len, maxPartsPerRow, minSplitSpan, numParts, numPoints)) {
          int32_t ps = 0, pe = 0;
          ringBounds(data, false, localPart, numParts, numPoints, ps, pe);
          char const* xy =
              data + kEmptyPolygonBlobSize + static_cast<std::size_t>(numParts) * 4;
          if (pe > ps) {
            double lminX = 1e308, lminY = 1e308;
            double lmaxX = -1e308, lmaxY = -1e308;
            for (int32_t p = ps; p < pe; ++p) {
              double const x = xyBytesX(xy, p);
              double const y = xyBytesY(xy, p);
              if (isnan(x) || isnan(y)) {
                continue;
              }
              lminX = fmin(lminX, x);
              lminY = fmin(lminY, y);
              lmaxX = fmax(lmaxX, x);
              lmaxY = fmax(lmaxY, y);
            }
            if (lmaxX >= lminX && lmaxY >= lminY) {
              bMinX = lminX;
              bMinY = lminY;
              bMaxX = lmaxX;
              bMaxY = lmaxY;
              ok = true;
            }
          }
        } else {
          ok = readEnvelopeFromBlob(
              data, len, bMinX, bMinY, bMaxX, bMaxY, invalidTypeFlag);
        }

        auto const idx = static_cast<cudf::size_type>(k);
        if (!ok) {
          cudf::clear_bit_unsafe(outMask, idx);
          return;
        }
        minX[idx] = bMinX - radius;
        minY[idx] = bMinY - radius;
        maxX[idx] = bMaxX + radius;
        maxY[idx] = bMaxY + radius;
      });

  auto const nullCount = cudf::null_count(
      out.envelopes.minX->view().null_mask(),
      0,
      static_cast<cudf::size_type>(total),
      stream);
  out.envelopes.minX->set_null_count(nullCount);
  auto maskBytes =
      cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(total));
  for (auto* col :
       {out.envelopes.minY.get(),
        out.envelopes.maxX.get(),
        out.envelopes.maxY.get()}) {
    auto dst =
        static_cast<cudf::bitmask_type*>(col->mutable_view().null_mask());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        dst,
        out.envelopes.minX->view().null_mask(),
        maskBytes,
        cudaMemcpyDeviceToDevice,
        stream.value()));
    col->set_null_count(nullCount);
  }
  return out;
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
dedupIndexPairs(
    std::unique_ptr<cudf::column> probeIdx,
    std::unique_ptr<cudf::column> buildIdx,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const total = probeIdx->size();
  if (total <= 1) {
    return {std::move(probeIdx), std::move(buildIdx)};
  }
  LOG_FIRST_N(WARNING, 20) << "[geo:dedup] pairs=" << total;
  auto* probeOut = probeIdx->mutable_view().data<cudf::size_type>();
  auto* buildOut = buildIdx->mutable_view().data<cudf::size_type>();
  auto zipIn =
      thrust::make_zip_iterator(thrust::make_tuple(probeOut, buildOut));
  thrust::sort(rmm::exec_policy(stream), zipIn, zipIn + total);
  auto zipEnd =
      thrust::unique(rmm::exec_policy(stream), zipIn, zipIn + total);
  auto const uniqueCount =
      static_cast<cudf::size_type>(thrust::distance(zipIn, zipEnd));
  if (uniqueCount == total) {
    return {std::move(probeIdx), std::move(buildIdx)};
  }
  auto probeUnique = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      uniqueCount,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto buildUnique = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      uniqueCount,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  thrust::copy_n(
      rmm::exec_policy(stream),
      probeOut,
      uniqueCount,
      probeUnique->mutable_view().data<cudf::size_type>());
  thrust::copy_n(
      rmm::exec_policy(stream),
      buildOut,
      uniqueCount,
      buildUnique->mutable_view().data<cudf::size_type>());
  return {std::move(probeUnique), std::move(buildUnique)};
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
geometryEnvelopeCrossIntersectIndices(
    GeometryEnvelopes const& probe,
    GeometryEnvelopes const& build,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const numProbe = probe.minX->size();
  auto const numBuild = build.minX->size();
  if (numProbe == 0 || numBuild == 0) {
    return {
        cudf::make_empty_column(cudf::type_to_id<cudf::size_type>()),
        cudf::make_empty_column(cudf::type_to_id<cudf::size_type>())};
  }
  auto const total =
      static_cast<int64_t>(numProbe) * static_cast<int64_t>(numBuild);
  CUDF_EXPECTS(
      total <= std::numeric_limits<cudf::size_type>::max(),
      "envelope cross product exceeds cudf::size_type");

  rmm::device_uvector<uint8_t> flags(static_cast<size_t>(total), stream, mr);
  auto* flagsPtr = flags.data();
  auto pMinX = probe.minX->view().data<double>();
  auto pMinY = probe.minY->view().data<double>();
  auto pMaxX = probe.maxX->view().data<double>();
  auto pMaxY = probe.maxY->view().data<double>();
  auto pNull = probe.minX->view().null_mask();
  auto bMinX = build.minX->view().data<double>();
  auto bMinY = build.minY->view().data<double>();
  auto bMaxX = build.maxX->view().data<double>();
  auto bMaxY = build.maxY->view().data<double>();
  auto bNull = build.minX->view().null_mask();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<int64_t>(0),
      total,
      [flagsPtr,
       numBuild,
       pMinX,
       pMinY,
       pMaxX,
       pMaxY,
       pNull,
       bMinX,
       bMinY,
       bMaxX,
       bMaxY,
       bNull] __device__(int64_t idx) {
        auto const pi = static_cast<cudf::size_type>(idx / numBuild);
        auto const bj = static_cast<cudf::size_type>(idx % numBuild);
        if ((pNull && !cudf::bit_is_set(pNull, pi)) ||
            (bNull && !cudf::bit_is_set(bNull, bj))) {
          flagsPtr[idx] = 0;
          return;
        }
        flagsPtr[idx] = envelopesIntersect(
                            pMinX[pi],
                            pMinY[pi],
                            pMaxX[pi],
                            pMaxY[pi],
                            bMinX[bj],
                            bMinY[bj],
                            bMaxX[bj],
                            bMaxY[bj])
            ? 1
            : 0;
      });

  rmm::device_uvector<cudf::size_type> offsets(
      static_cast<size_t>(total) + 1, stream, mr);
  thrust::exclusive_scan(
      rmm::exec_policy(stream),
      flags.begin(),
      flags.end(),
      offsets.begin(),
      cudf::size_type{0});
  cudf::size_type matchCount = 0;
  {
    cudf::size_type lastOffset = 0;
    uint8_t lastFlag = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &lastOffset,
        offsets.data() + total - 1,
        sizeof(cudf::size_type),
        cudaMemcpyDeviceToHost,
        stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &lastFlag,
        flags.data() + total - 1,
        sizeof(uint8_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    matchCount = lastOffset + static_cast<cudf::size_type>(lastFlag);
  }

  auto probeIdx = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      matchCount,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto buildIdx = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      matchCount,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (matchCount == 0) {
    return {std::move(probeIdx), std::move(buildIdx)};
  }
  auto* probeOut = probeIdx->mutable_view().data<cudf::size_type>();
  auto* buildOut = buildIdx->mutable_view().data<cudf::size_type>();
  auto* offsetsPtr = offsets.data();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<int64_t>(0),
      total,
      [flagsPtr, offsetsPtr, numBuild, probeOut, buildOut] __device__(
          int64_t idx) {
        if (flagsPtr[idx] == 0) {
          return;
        }
        auto const outPos = offsetsPtr[idx];
        probeOut[outPos] = static_cast<cudf::size_type>(idx / numBuild);
        buildOut[outPos] = static_cast<cudf::size_type>(idx % numBuild);
      });

  return {std::move(probeIdx), std::move(buildIdx)};
}

namespace {

__device__ inline void clampCellRange(
    double minV,
    double maxV,
    double origin,
    double invCell,
    int32_t nCells,
    int32_t& c0,
    int32_t& c1) {
  if (!isfinite(minV) || !isfinite(maxV) || minV > maxV || nCells <= 0) {
    // Empty range — callers must skip when c1 < c0.
    c0 = 0;
    c1 = -1;
    return;
  }
  c0 = static_cast<int32_t>(floor((minV - origin) * invCell));
  c1 = static_cast<int32_t>(floor((maxV - origin) * invCell));
  if (c0 < 0) {
    c0 = 0;
  }
  if (c1 < 0) {
    c1 = 0;
  }
  if (c0 >= nCells) {
    c0 = nCells - 1;
  }
  if (c1 >= nCells) {
    c1 = nCells - 1;
  }
  if (c1 < c0) {
    int32_t tmp = c0;
    c0 = c1;
    c1 = tmp;
  }
}

} // namespace

GeometryEnvelopeGrid buildGeometryEnvelopeGrid(
    GeometryEnvelopes buildEnvelopes,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  GeometryEnvelopeGrid grid;
  grid.envelopes = std::move(buildEnvelopes);
  auto const n = grid.envelopes.minX->size();
  auto emptyLarge = [&]() {
    grid.largeBuildIndices = cudf::make_empty_column(
        cudf::type_to_id<cudf::size_type>());
  };
  if (n == 0) {
    grid.nCols = 1;
    grid.nRows = 1;
    grid.cellOffsets = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT64},
        2,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    auto* off = grid.cellOffsets->mutable_view().data<int64_t>();
    thrust::fill_n(rmm::exec_policy(stream), off, 2, int64_t{0});
    grid.cellBuildIndices =
        cudf::make_empty_column(cudf::type_to_id<cudf::size_type>());
    emptyLarge();
    return grid;
  }

  auto minX = grid.envelopes.minX->view().data<double>();
  auto minY = grid.envelopes.minY->view().data<double>();
  auto maxX = grid.envelopes.maxX->view().data<double>();
  auto maxY = grid.envelopes.maxY->view().data<double>();
  auto nullMask = grid.envelopes.minX->view().null_mask();

  constexpr double kPosInf = 1.0e300;
  constexpr double kNegInf = -1.0e300;
  double hMinX = thrust::transform_reduce(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(n),
      [minX, nullMask] __device__(cudf::size_type i) -> double {
        if (nullMask && !cudf::bit_is_set(nullMask, i)) {
          return 1.0e300;
        }
        double const v = minX[i];
        return isnan(v) ? 1.0e300 : v;
      },
      kPosInf,
      thrust::minimum<double>());
  double hMinY = thrust::transform_reduce(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(n),
      [minY, nullMask] __device__(cudf::size_type i) -> double {
        if (nullMask && !cudf::bit_is_set(nullMask, i)) {
          return 1.0e300;
        }
        double const v = minY[i];
        return isnan(v) ? 1.0e300 : v;
      },
      kPosInf,
      thrust::minimum<double>());
  double hMaxX = thrust::transform_reduce(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(n),
      [maxX, nullMask] __device__(cudf::size_type i) -> double {
        if (nullMask && !cudf::bit_is_set(nullMask, i)) {
          return -1.0e300;
        }
        double const v = maxX[i];
        return isnan(v) ? -1.0e300 : v;
      },
      kNegInf,
      thrust::maximum<double>());
  double hMaxY = thrust::transform_reduce(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(n),
      [maxY, nullMask] __device__(cudf::size_type i) -> double {
        if (nullMask && !cudf::bit_is_set(nullMask, i)) {
          return -1.0e300;
        }
        double const v = maxY[i];
        return isnan(v) ? -1.0e300 : v;
      },
      kNegInf,
      thrust::maximum<double>());

  if (!(hMinX <= hMaxX) || !(hMinY <= hMaxY)) {
    grid.nCols = 1;
    grid.nRows = 1;
    grid.originX = 0;
    grid.originY = 0;
    grid.invCellW = 1;
    grid.invCellH = 1;
    grid.cellOffsets = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT64},
        2,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    auto* off = grid.cellOffsets->mutable_view().data<int64_t>();
    thrust::fill_n(rmm::exec_policy(stream), off, 2, int64_t{0});
    grid.cellBuildIndices =
        cudf::make_empty_column(cudf::type_to_id<cudf::size_type>());
    emptyLarge();
    return grid;
  }

  double const padX = (hMaxX - hMinX) * 1e-6 + 1e-12;
  double const padY = (hMaxY - hMinY) * 1e-6 + 1e-12;
  hMinX -= padX;
  hMaxX += padX;
  hMinY -= padY;
  hMaxY += padY;

  // A grid over polygon envelopes must stay coarse: each envelope is inserted
  // into every cell it overlaps, so fine cells explode the cell lists. Point
  // envelopes have zero extent and land in exactly one cell each, so that
  // risk does not apply and a coarse grid is actively harmful — 600M trip
  // pickups in a 128x128 grid leaves ~37k points per cell, and a single zone
  // probe then sweeps in hundreds of thousands of candidates (Q10).
  // cuSpatial's Morton quadtree similarly densifies around points; a uniform
  // 4096x4096 (~134MB of int64 offsets) is the flat-grid analogue that keeps
  // ~35 points/cell at SF100 while staying well under managed-memory budgets.
  double const maxEnvW = thrust::transform_reduce(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(n),
      [minX, maxX, nullMask] __device__(cudf::size_type i) -> double {
        if (nullMask && !cudf::bit_is_set(nullMask, i)) {
          return 0.0;
        }
        double const w = maxX[i] - minX[i];
        return isnan(w) ? 0.0 : w;
      },
      0.0,
      thrust::maximum<double>());
  double const maxEnvH = thrust::transform_reduce(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(n),
      [minY, maxY, nullMask] __device__(cudf::size_type i) -> double {
        if (nullMask && !cudf::bit_is_set(nullMask, i)) {
          return 0.0;
        }
        double const h = maxY[i] - minY[i];
        return isnan(h) ? 0.0 : h;
      },
      0.0,
      thrust::maximum<double>());
  grid.isPointGrid = (maxEnvW <= 0.0 && maxEnvH <= 0.0);
  // Bounded so cellOffsets stays manageable (4096^2 cells ~= 134MB of offsets).
  constexpr int32_t kMaxPolygonSide = 128;
  constexpr int32_t kMaxPointSide = 4096;
  int32_t const maxSide =
      grid.isPointGrid ? kMaxPointSide : kMaxPolygonSide;
  int32_t side =
      static_cast<int32_t>(std::ceil(std::sqrt(static_cast<double>(n))));
  side = std::max(16, std::min(side, maxSide));
  grid.nCols = side;
  grid.nRows = side;
  grid.originX = hMinX;
  grid.originY = hMinY;
  double const cellW = (hMaxX - hMinX) / grid.nCols;
  double const cellH = (hMaxY - hMinY) / grid.nRows;
  grid.invCellW = 1.0 / cellW;
  grid.invCellH = 1.0 / cellH;

  auto const nCells = static_cast<int64_t>(grid.nCols) * grid.nRows;
  rmm::device_uvector<int64_t> counts(static_cast<size_t>(nCells), stream, mr);
  thrust::fill_n(
      rmm::exec_policy(stream), counts.begin(), counts.size(), int64_t{0});
  auto* countsPtr = counts.data();
  int32_t const nCols = grid.nCols;
  int32_t const nRows = grid.nRows;
  double const originX = grid.originX;
  double const originY = grid.originY;
  double const invCellW = grid.invCellW;
  double const invCellH = grid.invCellH;
  int32_t const maxCells = GeometryEnvelopeGrid::kMaxCellsPerBuild;

  rmm::device_uvector<cudf::size_type> largeFlags(
      static_cast<size_t>(n), stream, mr);
  thrust::fill_n(
      rmm::exec_policy(stream), largeFlags.begin(), largeFlags.size(), 0);
  auto* largeFlagsPtr = largeFlags.data();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      n,
      [minX,
       minY,
       maxX,
       maxY,
       nullMask,
       countsPtr,
       largeFlagsPtr,
       nCols,
       nRows,
       originX,
       originY,
       invCellW,
       invCellH,
       maxCells] __device__(cudf::size_type i) {
        if (nullMask && !cudf::bit_is_set(nullMask, i)) {
          return;
        }
        int32_t c0, c1, r0, r1;
        clampCellRange(minX[i], maxX[i], originX, invCellW, nCols, c0, c1);
        clampCellRange(minY[i], maxY[i], originY, invCellH, nRows, r0, r1);
        if (c1 < c0 || r1 < r0) {
          return;
        }
        int64_t const span = static_cast<int64_t>(c1 - c0 + 1) *
            static_cast<int64_t>(r1 - r0 + 1);
        if (span > maxCells) {
          largeFlagsPtr[i] = 1;
          // Still index the centroid cell so point probes near the center
          // hit the grid path; large list covers the rest.
          int32_t const cc = c0 + (c1 - c0) / 2;
          int32_t const rr = r0 + (r1 - r0) / 2;
          atomicAdd(
              reinterpret_cast<unsigned long long*>(
                  &countsPtr[static_cast<int64_t>(rr) * nCols + cc]),
              1ULL);
          return;
        }
        for (int32_t r = r0; r <= r1; ++r) {
          for (int32_t c = c0; c <= c1; ++c) {
            atomicAdd(
                reinterpret_cast<unsigned long long*>(
                    &countsPtr[static_cast<int64_t>(r) * nCols + c]),
                1ULL);
          }
        }
      });

  cudf::size_type numLarge = static_cast<cudf::size_type>(thrust::reduce(
      rmm::exec_policy(stream),
      largeFlags.begin(),
      largeFlags.end(),
      cudf::size_type{0}));
  grid.largeBuildIndices = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      numLarge,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (numLarge > 0) {
    auto* largeOut =
        grid.largeBuildIndices->mutable_view().data<cudf::size_type>();
    thrust::copy_if(
        rmm::exec_policy(stream),
        thrust::counting_iterator<cudf::size_type>(0),
        thrust::counting_iterator<cudf::size_type>(n),
        largeFlags.begin(),
        largeOut,
        [] __device__(cudf::size_type flag) { return flag != 0; });
  }

  grid.cellOffsets = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64},
      static_cast<cudf::size_type>(nCells + 1),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* offsetsPtr = grid.cellOffsets->mutable_view().data<int64_t>();
  thrust::exclusive_scan(
      rmm::exec_policy(stream),
      counts.begin(),
      counts.end(),
      offsetsPtr,
      int64_t{0});
  int64_t totalEntries = 0;
  {
    int64_t lastCount = 0;
    int64_t lastOffset = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &lastCount,
        counts.data() + nCells - 1,
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &lastOffset,
        offsetsPtr + nCells - 1,
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    totalEntries = lastOffset + lastCount;
  }
  CUDF_EXPECTS(
      totalEntries <= static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
      "Spatial envelope grid cell list too large");
  LOG_FIRST_N(WARNING, 10) << "[geo:grid] n=" << n << " nCells=" << nCells
                           << " side=" << grid.nCols
                           << " pointGrid=" << grid.isPointGrid
                           << " totalEntries=" << totalEntries
                           << " numLarge=" << numLarge;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      offsetsPtr + nCells,
      &totalEntries,
      sizeof(int64_t),
      cudaMemcpyHostToDevice,
      stream.value()));

  grid.cellBuildIndices = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      static_cast<cudf::size_type>(totalEntries),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (totalEntries == 0) {
    return grid;
  }

  thrust::copy_n(
      rmm::exec_policy(stream), offsetsPtr, nCells, counts.begin());
  auto* insertPtr = counts.data();
  auto* cellBuildPtr =
      grid.cellBuildIndices->mutable_view().data<cudf::size_type>();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      n,
      [minX,
       minY,
       maxX,
       maxY,
       nullMask,
       insertPtr,
       cellBuildPtr,
       nCols,
       nRows,
       originX,
       originY,
       invCellW,
       invCellH,
       maxCells] __device__(cudf::size_type i) {
        if (nullMask && !cudf::bit_is_set(nullMask, i)) {
          return;
        }
        int32_t c0, c1, r0, r1;
        clampCellRange(minX[i], maxX[i], originX, invCellW, nCols, c0, c1);
        clampCellRange(minY[i], maxY[i], originY, invCellH, nRows, r0, r1);
        if (c1 < c0 || r1 < r0) {
          return;
        }
        int64_t const span = static_cast<int64_t>(c1 - c0 + 1) *
            static_cast<int64_t>(r1 - r0 + 1);
        if (span > maxCells) {
          int32_t const cc = c0 + (c1 - c0) / 2;
          int32_t const rr = r0 + (r1 - r0) / 2;
          auto const pos = static_cast<int64_t>(atomicAdd(
              reinterpret_cast<unsigned long long*>(
                  &insertPtr[static_cast<int64_t>(rr) * nCols + cc]),
              1ULL));
          cellBuildPtr[pos] = i;
          return;
        }
        for (int32_t r = r0; r <= r1; ++r) {
          for (int32_t c = c0; c <= c1; ++c) {
            auto const pos = static_cast<int64_t>(atomicAdd(
                reinterpret_cast<unsigned long long*>(
                    &insertPtr[static_cast<int64_t>(r) * nCols + c]),
                1ULL));
            cellBuildPtr[pos] = i;
          }
        }
      });

  return grid;
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
queryGeometryEnvelopeGrid(
    GeometryEnvelopeGrid const& grid,
    GeometryEnvelopes const& probe,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    int64_t maxMatches) {
  auto const numProbe = probe.minX->size();
  auto numLarge =
      grid.largeBuildIndices ? grid.largeBuildIndices->size() : 0;
  std::unique_ptr<cudf::column> allBuildFallback;
  // If the cell list is empty and no large list was produced, fall back to
  // testing every build envelope (correct but slower). Prevents silent
  // zero-match joins when indexing drops all rows.
  if (numProbe > 0 && grid.cellBuildIndices->size() == 0 && numLarge == 0 &&
      grid.envelopes.minX->size() > 0) {
    auto const nBuild = grid.envelopes.minX->size();
    allBuildFallback = cudf::make_numeric_column(
        cudf::data_type{cudf::type_to_id<cudf::size_type>()},
        nBuild,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    thrust::sequence(
        rmm::exec_policy(stream),
        allBuildFallback->mutable_view().begin<cudf::size_type>(),
        allBuildFallback->mutable_view().end<cudf::size_type>(),
        cudf::size_type{0});
    numLarge = nBuild;
  }
  if (numProbe == 0 ||
      (grid.cellBuildIndices->size() == 0 && numLarge == 0)) {
    return {
        cudf::make_empty_column(cudf::type_to_id<cudf::size_type>()),
        cudf::make_empty_column(cudf::type_to_id<cudf::size_type>())};
  }

  auto pMinX = probe.minX->view().data<double>();
  auto pMinY = probe.minY->view().data<double>();
  auto pMaxX = probe.maxX->view().data<double>();
  auto pMaxY = probe.maxY->view().data<double>();
  auto pNull = probe.minX->view().null_mask();
  auto bMinX = grid.envelopes.minX->view().data<double>();
  auto bMinY = grid.envelopes.minY->view().data<double>();
  auto bMaxX = grid.envelopes.maxX->view().data<double>();
  auto bMaxY = grid.envelopes.maxY->view().data<double>();
  auto bNull = grid.envelopes.minX->view().null_mask();
  auto cellOffsets = grid.cellOffsets->view().data<int64_t>();
  auto cellBuilds =
      grid.cellBuildIndices->view().data<cudf::size_type>();
  cudf::size_type const* largeBuilds = nullptr;
  if (allBuildFallback) {
    largeBuilds = allBuildFallback->view().data<cudf::size_type>();
  } else if (numLarge > 0) {
    largeBuilds = grid.largeBuildIndices->view().data<cudf::size_type>();
  }
  int32_t const nCols = grid.nCols;
  int32_t const nRows = grid.nRows;
  double const originX = grid.originX;
  double const originY = grid.originY;
  double const invCellW = grid.invCellW;
  double const invCellH = grid.invCellH;

  rmm::device_uvector<int64_t> matchCounts(
      static_cast<size_t>(numProbe), stream, mr);
  auto* matchCountsPtr = matchCounts.data();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      numProbe,
      [pMinX,
       pMinY,
       pMaxX,
       pMaxY,
       pNull,
       bMinX,
       bMinY,
       bMaxX,
       bMaxY,
       bNull,
       cellOffsets,
       cellBuilds,
       largeBuilds,
       numLarge,
       nCols,
       nRows,
       originX,
       originY,
       invCellW,
       invCellH,
       matchCountsPtr] __device__(cudf::size_type pi) {
        if (pNull && !cudf::bit_is_set(pNull, pi)) {
          matchCountsPtr[pi] = 0;
          return;
        }
        int64_t count = 0;
        int32_t c0, c1, r0, r1;
        clampCellRange(
            pMinX[pi], pMaxX[pi], originX, invCellW, nCols, c0, c1);
        clampCellRange(
            pMinY[pi], pMaxY[pi], originY, invCellH, nRows, r0, r1);
        if (c1 >= c0 && r1 >= r0) {
          for (int32_t r = r0; r <= r1; ++r) {
            for (int32_t c = c0; c <= c1; ++c) {
              int64_t const cell = static_cast<int64_t>(r) * nCols + c;
              int64_t const begin = cellOffsets[cell];
              int64_t const end = cellOffsets[cell + 1];
              for (int64_t k = begin; k < end; ++k) {
                auto const bj = cellBuilds[k];
                if (bNull && !cudf::bit_is_set(bNull, bj)) {
                  continue;
                }
                if (envelopesIntersect(
                        pMinX[pi],
                        pMinY[pi],
                        pMaxX[pi],
                        pMaxY[pi],
                        bMinX[bj],
                        bMinY[bj],
                        bMaxX[bj],
                        bMaxY[bj])) {
                  ++count;
                }
              }
            }
          }
        }
        for (cudf::size_type li = 0; li < numLarge; ++li) {
          auto const bj = largeBuilds[li];
          if (bNull && !cudf::bit_is_set(bNull, bj)) {
            continue;
          }
          if (envelopesIntersect(
                  pMinX[pi],
                  pMinY[pi],
                  pMaxX[pi],
                  pMaxY[pi],
                  bMinX[bj],
                  bMinY[bj],
                  bMaxX[bj],
                  bMaxY[bj])) {
            ++count;
          }
        }
        matchCountsPtr[pi] = count;
      });

  rmm::device_uvector<int64_t> probeOffsets(
      static_cast<size_t>(numProbe) + 1, stream, mr);
  thrust::exclusive_scan(
      rmm::exec_policy(stream),
      matchCounts.begin(),
      matchCounts.end(),
      probeOffsets.begin(),
      int64_t{0});
  cudf::size_type totalMatches = 0;
  {
    int64_t lastCount = 0;
    int64_t lastOffset = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &lastCount,
        matchCounts.data() + numProbe - 1,
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &lastOffset,
        probeOffsets.data() + numProbe - 1,
        sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    int64_t const total64 = lastOffset + lastCount;
    int64_t const budget = std::min<int64_t>(
        maxMatches,
        static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()));
    LOG_FIRST_N(WARNING, 20)
        << "[geo:query] numProbe=" << numProbe << " numLarge=" << numLarge
        << " total=" << total64 << " budget=" << budget;
    CUDF_EXPECTS(
        total64 <= budget,
        "Spatial envelope grid match count overflow; reduce probe batch size");
    totalMatches = static_cast<cudf::size_type>(total64);
  }

  auto probeIdx = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      totalMatches,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto buildIdx = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      totalMatches,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (totalMatches == 0) {
    return {std::move(probeIdx), std::move(buildIdx)};
  }

  auto* probeOut = probeIdx->mutable_view().data<cudf::size_type>();
  auto* buildOut = buildIdx->mutable_view().data<cudf::size_type>();
  auto* probeOffsetsPtr = probeOffsets.data();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      numProbe,
      [pMinX,
       pMinY,
       pMaxX,
       pMaxY,
       pNull,
       bMinX,
       bMinY,
       bMaxX,
       bMaxY,
       bNull,
       cellOffsets,
       cellBuilds,
       largeBuilds,
       numLarge,
       nCols,
       nRows,
       originX,
       originY,
       invCellW,
       invCellH,
       probeOffsetsPtr,
       probeOut,
       buildOut] __device__(cudf::size_type pi) {
        if (pNull && !cudf::bit_is_set(pNull, pi)) {
          return;
        }
        int64_t outPos = probeOffsetsPtr[pi];
        int32_t c0, c1, r0, r1;
        clampCellRange(
            pMinX[pi], pMaxX[pi], originX, invCellW, nCols, c0, c1);
        clampCellRange(
            pMinY[pi], pMaxY[pi], originY, invCellH, nRows, r0, r1);
        if (c1 >= c0 && r1 >= r0) {
          for (int32_t r = r0; r <= r1; ++r) {
            for (int32_t c = c0; c <= c1; ++c) {
              int64_t const cell = static_cast<int64_t>(r) * nCols + c;
              int64_t const begin = cellOffsets[cell];
              int64_t const end = cellOffsets[cell + 1];
              for (int64_t k = begin; k < end; ++k) {
                auto const bj = cellBuilds[k];
                if (bNull && !cudf::bit_is_set(bNull, bj)) {
                  continue;
                }
                if (envelopesIntersect(
                        pMinX[pi],
                        pMinY[pi],
                        pMaxX[pi],
                        pMaxY[pi],
                        bMinX[bj],
                        bMinY[bj],
                        bMaxX[bj],
                        bMaxY[bj])) {
                  probeOut[outPos] = pi;
                  buildOut[outPos] = bj;
                  ++outPos;
                }
              }
            }
          }
        }
        for (cudf::size_type li = 0; li < numLarge; ++li) {
          auto const bj = largeBuilds[li];
          if (bNull && !cudf::bit_is_set(bNull, bj)) {
            continue;
          }
          if (envelopesIntersect(
                  pMinX[pi],
                  pMinY[pi],
                  pMaxX[pi],
                  pMaxY[pi],
                  bMinX[bj],
                  bMinY[bj],
                  bMaxX[bj],
                  bMaxY[bj])) {
            probeOut[outPos] = pi;
            buildOut[outPos] = bj;
            ++outPos;
          }
        }
      });

  // Collapse duplicate pairs from multi-cell builds. Point grids never
  // produce duplicates (each point is in exactly one cell, and largeBuild
  // is empty), so skip the expensive sort+unique — cuSpatial's quadtree
  // leaves have the same uniqueness property.
  if (totalMatches > 1 && !grid.isPointGrid) {
    auto zipIn = thrust::make_zip_iterator(
        thrust::make_tuple(probeOut, buildOut));
    thrust::sort(rmm::exec_policy(stream), zipIn, zipIn + totalMatches);
    auto zipEnd = thrust::unique(
        rmm::exec_policy(stream), zipIn, zipIn + totalMatches);
    auto uniqueCount = static_cast<cudf::size_type>(
        thrust::distance(zipIn, zipEnd));
    if (uniqueCount < totalMatches) {
      auto probeUnique = cudf::make_numeric_column(
          cudf::data_type{cudf::type_to_id<cudf::size_type>()},
          uniqueCount,
          cudf::mask_state::UNALLOCATED,
          stream,
          mr);
      auto buildUnique = cudf::make_numeric_column(
          cudf::data_type{cudf::type_to_id<cudf::size_type>()},
          uniqueCount,
          cudf::mask_state::UNALLOCATED,
          stream,
          mr);
      thrust::copy_n(
          rmm::exec_policy(stream),
          probeOut,
          uniqueCount,
          probeUnique->mutable_view().data<cudf::size_type>());
      thrust::copy_n(
          rmm::exec_policy(stream),
          buildOut,
          uniqueCount,
          buildUnique->mutable_view().data<cudf::size_type>());
      return {std::move(probeUnique), std::move(buildUnique)};
    }
  }

  return {std::move(probeIdx), std::move(buildIdx)};
}

namespace {

constexpr int32_t kMaxKnnK = 32;

__device__ inline void knnTryInsert(
    double* topDist,
    cudf::size_type* topBuild,
    int32_t k,
    cudf::size_type buildIdx,
    double dist) {
  // Already present?
  for (int32_t i = 0; i < k; ++i) {
    if (topBuild[i] == buildIdx) {
      return;
    }
  }
  // Find worst (max dist) slot; empty slots use +inf / -1.
  int32_t worst = 0;
  for (int32_t i = 1; i < k; ++i) {
    if (topDist[i] > topDist[worst]) {
      worst = i;
    }
  }
  if (dist < topDist[worst]) {
    topDist[worst] = dist;
    topBuild[worst] = buildIdx;
  }
}

} // namespace

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
geometryKnnJoinIndices(
    cudf::column_view const& probeGeometry,
    cudf::column_view const& buildGeometry,
    GeometryEnvelopeGrid const& buildGrid,
    int32_t knnK,
    int32_t* invalidTypeFlag,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      probeGeometry.type().id() == cudf::type_id::STRING &&
          buildGeometry.type().id() == cudf::type_id::STRING,
      "ST_KNN expects geometry/STRING columns");
  CUDF_EXPECTS(knnK >= 1 && knnK <= kMaxKnnK, "ST_KNN k out of range");

  auto const numProbe = probeGeometry.size();
  auto const numBuild = buildGeometry.size();
  if (numProbe == 0 || numBuild == 0) {
    return {
        cudf::make_empty_column(cudf::type_to_id<cudf::size_type>()),
        cudf::make_empty_column(cudf::type_to_id<cudf::size_type>())};
  }

  int32_t const k = std::min<int32_t>(knnK, numBuild);

  // Per-probe top-k heaps (row-major: probe * k + slot).
  rmm::device_uvector<double> topDist(
      static_cast<std::size_t>(numProbe) * static_cast<std::size_t>(k),
      stream,
      mr);
  rmm::device_uvector<cudf::size_type> topBuild(
      static_cast<std::size_t>(numProbe) * static_cast<std::size_t>(k),
      stream,
      mr);
  thrust::fill(
      rmm::exec_policy(stream),
      topDist.begin(),
      topDist.end(),
      std::numeric_limits<double>::infinity());
  thrust::fill(
      rmm::exec_policy(stream),
      topBuild.begin(),
      topBuild.end(),
      static_cast<cudf::size_type>(-1));

  auto emptyExpand = cudf::make_empty_column(cudf::type_id::FLOAT64);
  // Expanding search radii (degrees). Stop per-probe when k-th dist <= radius.
  static constexpr double kRadii[] = {
      0.005, 0.02, 0.1, 0.5, 2.0, 20.0, 180.0};
  constexpr int32_t kNumRadii = 7;

  cudf::strings_column_view probeStr(probeGeometry);
  cudf::strings_column_view buildStr(buildGeometry);
  auto probeChars = probeStr.chars_begin(stream);
  auto buildChars = buildStr.chars_begin(stream);
  auto probeOff =
      cudf::detail::offsetalator_factory::make_input_iterator(probeStr.offsets());
  auto buildOff =
      cudf::detail::offsetalator_factory::make_input_iterator(buildStr.offsets());
  auto probeNull = probeGeometry.null_mask();
  auto buildNull = buildGeometry.null_mask();
  auto* topDistPtr = topDist.data();
  auto* topBuildPtr = topBuild.data();

  // Tracks probes that still need a larger search radius.
  rmm::device_uvector<uint8_t> probeNeedsExpand(
      static_cast<std::size_t>(numProbe), stream, mr);
  thrust::fill(
      rmm::exec_policy(stream),
      probeNeedsExpand.begin(),
      probeNeedsExpand.end(),
      uint8_t{1});
  auto* needsExpandPtr = probeNeedsExpand.data();

  for (int32_t ri = 0; ri < kNumRadii; ++ri) {
    double const radius = kRadii[ri];
    auto probeEnv = extractGeometryEnvelopes(
        probeGeometry,
        emptyExpand->view(),
        /*constantExpandBy=*/radius,
        invalidTypeFlag,
        stream,
        mr);

    // Finished probes: empty envelopes so they contribute 0 grid matches.
    {
      auto* pMinX = probeEnv.minX->mutable_view().data<double>();
      auto* pMaxX = probeEnv.maxX->mutable_view().data<double>();
      auto* pMinY = probeEnv.minY->mutable_view().data<double>();
      auto* pMaxY = probeEnv.maxY->mutable_view().data<double>();
      thrust::for_each_n(
          rmm::exec_policy(stream),
          thrust::counting_iterator<cudf::size_type>(0),
          numProbe,
          [needsExpandPtr, pMinX, pMaxX, pMinY, pMaxY] __device__(
              cudf::size_type pi) {
            if (!needsExpandPtr[pi]) {
              pMinX[pi] = 1.0;
              pMaxX[pi] = 0.0;
              pMinY[pi] = 1.0;
              pMaxY[pi] = 0.0;
            }
          });
    }

    // Budget must fit batch×build in the worst case (full-world radius).
    int64_t const kMaxCand = std::min<int64_t>(
        64'000'000,
        std::max<int64_t>(
            static_cast<int64_t>(numProbe) *
                static_cast<int64_t>(numBuild),
            1));
    auto [candProbe, candBuild] = queryGeometryEnvelopeGrid(
        buildGrid, probeEnv, stream, mr, kMaxCand);
    auto const nCand = candProbe->size();
    if (nCand == 0) {
      // No candidates at this radius for unfinished probes; expand further.
    } else {
      auto* candP = candProbe->mutable_view().data<cudf::size_type>();
      auto* candB = candBuild->mutable_view().data<cudf::size_type>();

      // Group candidates by probe so each probe's top-k update is serial.
      auto zip = thrust::make_zip_iterator(thrust::make_tuple(candP, candB));
      thrust::sort(
          rmm::exec_policy(stream), zip, zip + nCand, [] __device__(auto a, auto b) {
            return thrust::get<0>(a) < thrust::get<0>(b);
          });

      rmm::device_uvector<cudf::size_type> probeBegin(
          static_cast<std::size_t>(numProbe) + 1, stream, mr);
      thrust::fill(
          rmm::exec_policy(stream),
          probeBegin.begin(),
          probeBegin.end(),
          nCand);
      auto* probeBeginPtr = probeBegin.data();
      thrust::for_each_n(
          rmm::exec_policy(stream),
          thrust::counting_iterator<cudf::size_type>(0),
          nCand,
          [candP, probeBeginPtr] __device__(cudf::size_type i) {
            auto const pi = candP[i];
            atomicMin(probeBeginPtr + pi, i);
          });
      // probeEnd[pi] = probeBegin[next probe with candidates] or nCand.
      rmm::device_uvector<cudf::size_type> probeEnd(
          static_cast<std::size_t>(numProbe), stream, mr);
      thrust::fill(
          rmm::exec_policy(stream), probeEnd.begin(), probeEnd.end(), nCand);
      auto* probeEndPtr = probeEnd.data();
      thrust::for_each_n(
          rmm::exec_policy(stream),
          thrust::counting_iterator<cudf::size_type>(0),
          nCand,
          [candP, probeEndPtr] __device__(cudf::size_type i) {
            auto const pi = candP[i];
            atomicMax(probeEndPtr + pi, i + 1);
          });

      thrust::for_each_n(
          rmm::exec_policy(stream),
          thrust::counting_iterator<cudf::size_type>(0),
          numProbe,
          [candP,
           candB,
           probeBeginPtr,
           probeEndPtr,
           probeChars,
           buildChars,
           probeOff,
           buildOff,
           probeNull,
           buildNull,
           topDistPtr,
           topBuildPtr,
           needsExpandPtr,
           invalidTypeFlag,
           k,
           nCand,
           probeNullCount = probeGeometry.null_count(),
           buildNullCount = buildGeometry.null_count()] __device__(
              cudf::size_type pi) {
            if (!needsExpandPtr[pi]) {
              return;
            }
            if (probeNullCount > 0 && probeNull != nullptr &&
                !cudf::bit_is_set(probeNull, pi)) {
              return;
            }
            auto const begin = probeBeginPtr[pi];
            auto const end = probeEndPtr[pi];
            if (begin >= nCand || begin >= end) {
              return;
            }
            for (cudf::size_type i = begin; i < end; ++i) {
              if (candP[i] != pi) {
                continue;
              }
              auto const bi = candB[i];
              if (buildNullCount > 0 && buildNull != nullptr &&
                  !cudf::bit_is_set(buildNull, bi)) {
                continue;
              }
              auto const ps = probeOff[pi];
              auto const pe = probeOff[pi + 1];
              auto const bs = buildOff[bi];
              auto const be = buildOff[bi + 1];
              char const* pData = probeChars + ps;
              char const* bData = buildChars + bs;
              auto const pLen = static_cast<cudf::size_type>(pe - ps);
              auto const bLen = static_cast<cudf::size_type>(be - bs);
              if (pLen < 1 || bLen < 1) {
                continue;
              }
              uint8_t const pTag = static_cast<uint8_t>(pData[0]);
              uint8_t const bTag = static_cast<uint8_t>(bData[0]);

              double dist = 0;
              if (pTag == kPointTag && bTag == kPointTag) {
                double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
                if (!readPointXY(pData, pLen, x1, y1, invalidTypeFlag) ||
                    !readPointXY(bData, bLen, x2, y2, invalidTypeFlag)) {
                  continue;
                }
                if (isEmptyPoint(x1, y1) || isEmptyPoint(x2, y2)) {
                  continue;
                }
                double const dx = x1 - x2;
                double const dy = y1 - y2;
                dist = sqrt(dx * dx + dy * dy);
              } else if (pTag == kPointTag) {
                double px = 0, py = 0;
                if (!readPointXY(pData, pLen, px, py, invalidTypeFlag) ||
                    isEmptyPoint(px, py)) {
                  continue;
                }
                if (!distPointToPolygonBlob(
                        px, py, bData, bLen, dist, invalidTypeFlag)) {
                  continue;
                }
              } else if (bTag == kPointTag) {
                double px = 0, py = 0;
                if (!readPointXY(bData, bLen, px, py, invalidTypeFlag) ||
                    isEmptyPoint(px, py)) {
                  continue;
                }
                if (!distPointToPolygonBlob(
                        px, py, pData, pLen, dist, invalidTypeFlag)) {
                  continue;
                }
              } else {
                continue;
              }
              knnTryInsert(
                  topDistPtr + static_cast<std::size_t>(pi) * k,
                  topBuildPtr + static_cast<std::size_t>(pi) * k,
                  k,
                  bi,
                  dist);
            }
          });
    }

    // Done when every non-null probe has k neighbors with worst dist <= radius.
    rmm::device_scalar<int32_t> unfinished(0, stream, mr);
    auto* unfinishedPtr = unfinished.data();
    thrust::for_each_n(
        rmm::exec_policy(stream),
        thrust::counting_iterator<cudf::size_type>(0),
        numProbe,
        [topDistPtr,
         topBuildPtr,
         unfinishedPtr,
         needsExpandPtr,
         probeNull,
         k,
         radius,
         probeNullCount = probeGeometry.null_count()] __device__(
            cudf::size_type pi) {
          if (probeNullCount > 0 && probeNull != nullptr &&
              !cudf::bit_is_set(probeNull, pi)) {
            needsExpandPtr[pi] = 0;
            return;
          }
          double worst = -1.0;
          int32_t filled = 0;
          for (int32_t s = 0; s < k; ++s) {
            auto const bi =
                topBuildPtr[static_cast<std::size_t>(pi) * k + s];
            auto const d =
                topDistPtr[static_cast<std::size_t>(pi) * k + s];
            if (bi >= 0 && isfinite(d)) {
              ++filled;
              worst = fmax(worst, d);
            }
          }
          if (filled < k || worst > radius) {
            needsExpandPtr[pi] = 1;
            atomicAdd(unfinishedPtr, 1);
          } else {
            needsExpandPtr[pi] = 0;
          }
        });
    int32_t unfinishedHost = unfinished.value(stream);
    stream.synchronize();
    if (unfinishedHost == 0) {
      break;
    }
  }

  // Compact heaps → index pairs.
  rmm::device_uvector<cudf::size_type> sizes(
      static_cast<std::size_t>(numProbe) + 1, stream, mr);
  auto* sizesPtr = sizes.data();
  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      numProbe + 1,
      [sizesPtr, topBuildPtr, k, numProbe] __device__(cudf::size_type i) {
        if (i == numProbe) {
          sizesPtr[i] = 0;
          return;
        }
        cudf::size_type n = 0;
        for (int32_t s = 0; s < k; ++s) {
          if (topBuildPtr[static_cast<std::size_t>(i) * k + s] >= 0) {
            ++n;
          }
        }
        sizesPtr[i] = n;
      });

  auto offsetsCol = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      numProbe + 1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* offsets = offsetsCol->mutable_view().data<cudf::size_type>();
  thrust::exclusive_scan(
      rmm::exec_policy(stream), sizes.begin(), sizes.end(), offsets);

  cudf::size_type totalPairs = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &totalPairs,
      offsets + numProbe,
      sizeof(cudf::size_type),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  auto probeIdx = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      totalPairs,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto buildIdx = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<cudf::size_type>()},
      totalPairs,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto* probeOut = probeIdx->mutable_view().data<cudf::size_type>();
  auto* buildOut = buildIdx->mutable_view().data<cudf::size_type>();

  thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      numProbe,
      [probeOut,
       buildOut,
       offsets,
       topBuildPtr,
       topDistPtr,
       k] __device__(cudf::size_type pi) {
        // Emit in ascending distance order for determinism.
        double dists[kMaxKnnK];
        cudf::size_type builds[kMaxKnnK];
        int32_t n = 0;
        for (int32_t s = 0; s < k; ++s) {
          auto const bi =
              topBuildPtr[static_cast<std::size_t>(pi) * k + s];
          auto const d =
              topDistPtr[static_cast<std::size_t>(pi) * k + s];
          if (bi >= 0 && isfinite(d)) {
            dists[n] = d;
            builds[n] = bi;
            ++n;
          }
        }
        // Insertion sort by distance then buildIdx.
        for (int32_t i = 1; i < n; ++i) {
          double const d = dists[i];
          cudf::size_type const b = builds[i];
          int32_t j = i - 1;
          while (j >= 0 &&
                 (dists[j] > d || (dists[j] == d && builds[j] > b))) {
            dists[j + 1] = dists[j];
            builds[j + 1] = builds[j];
            --j;
          }
          dists[j + 1] = d;
          builds[j + 1] = b;
        }
        auto out = offsets[pi];
        for (int32_t i = 0; i < n; ++i) {
          probeOut[out + i] = pi;
          buildOut[out + i] = builds[i];
        }
      });

  return {std::move(probeIdx), std::move(buildIdx)};
}

} // namespace facebook::velox::cudf_velox
