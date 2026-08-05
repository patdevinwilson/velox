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

} // namespace facebook::velox::cudf_velox
