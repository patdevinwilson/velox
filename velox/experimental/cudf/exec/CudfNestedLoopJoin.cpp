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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfNestedLoopJoin.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/expression/GeometryKernels.h"
#include "velox/experimental/cudf/expression/PrecomputeInstruction.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Task.h"
#include "velox/core/Expressions.h"
#include "velox/type/TypeUtil.h"
#include "velox/vector/SimpleVector.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_scalar.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {

namespace {

// Appends precomputed columns to a table view for filter AST evaluation.
// TODO: Consolidate with the identical helper in CudfHashJoin.cpp.
cudf::table_view createExtendedTableView(
    cudf::table_view originalView,
    std::vector<ColumnOrView>& precomputedColumns) {
  if (precomputedColumns.empty()) {
    return originalView;
  }
  std::vector<cudf::column_view> allViews;
  allViews.reserve(originalView.num_columns() + precomputedColumns.size());
  for (cudf::size_type i = 0; i < originalView.num_columns(); ++i) {
    allViews.push_back(originalView.column(i));
  }
  for (auto& col : precomputedColumns) {
    allViews.push_back(asView(col));
  }
  return cudf::table_view(allViews);
}

bool endsWithIgnoreCase(std::string_view s, std::string_view suffix) {
  if (s.size() < suffix.size()) {
    return false;
  }
  auto tail = s.substr(s.size() - suffix.size());
  for (size_t i = 0; i < suffix.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(tail[i])) !=
        std::tolower(static_cast<unsigned char>(suffix[i]))) {
      return false;
    }
  }
  return true;
}

/// Unwrap ST_GeomFromBinary(field) or bare field → (field name, isWkb).
std::optional<std::pair<std::string, bool>> geometryFieldRef(
    const core::TypedExprPtr& expr) {
  if (!expr) {
    return std::nullopt;
  }
  if (auto field =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr)) {
    return std::make_pair(field->name(), false);
  }
  if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expr)) {
    if (endsWithIgnoreCase(call->name(), "st_geomfrombinary") &&
        call->inputs().size() == 1) {
      auto inner = geometryFieldRef(call->inputs()[0]);
      if (!inner) {
        return std::nullopt;
      }
      return std::make_pair(inner->first, true);
    }
  }
  return std::nullopt;
}

/// Collect leaf conjuncts under AND (and nested ANDs).
void collectAndConjuncts(
    const core::TypedExprPtr& expr,
    std::vector<core::TypedExprPtr>& out) {
  auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expr);
  if (call && endsWithIgnoreCase(call->name(), "and") &&
      call->inputs().size() >= 2) {
    for (auto const& in : call->inputs()) {
      collectAndConjuncts(in, out);
    }
    return;
  }
  out.push_back(expr);
}

/// Best-effort axis-aligned envelope of a constant geometry / WKT string.
std::optional<std::array<double, 4>> constantGeometryAabb(
    const core::TypedExprPtr& expr) {
  if (!expr) {
    return std::nullopt;
  }

  auto aabbFromXy = [](std::vector<double> const& xy)
      -> std::optional<std::array<double, 4>> {
    if (xy.size() < 4) {
      return std::nullopt;
    }
    double minX = xy[0], maxX = xy[0], minY = xy[1], maxY = xy[1];
    for (size_t i = 0; i + 1 < xy.size(); i += 2) {
      minX = std::min(minX, xy[i]);
      maxX = std::max(maxX, xy[i]);
      minY = std::min(minY, xy[i + 1]);
      maxY = std::max(maxY, xy[i + 1]);
    }
    return std::array<double, 4>{minX, minY, maxX, maxY};
  };

  auto aabbFromBytes = [&](std::string_view bytes)
      -> std::optional<std::array<double, 4>> {
    if (bytes.empty()) {
      return std::nullopt;
    }
    // Velox geometry blob (tag byte first).
    std::vector<double> xy;
    std::vector<int32_t> partEnds;
    if (parseVeloxPolygon(bytes, xy, partEnds)) {
      return aabbFromXy(xy);
    }
    // WKT POLYGON((x y, ...)) — scan numeric tokens.
    double minX = 0, maxX = 0, minY = 0, maxY = 0;
    bool any = false;
    const char* p = bytes.data();
    const char* end = p + bytes.size();
    while (p < end) {
      while (p < end &&
             !(std::isdigit(static_cast<unsigned char>(*p)) || *p == '-' ||
               *p == '+' || *p == '.')) {
        ++p;
      }
      if (p >= end) {
        break;
      }
      char* next = nullptr;
      double v = std::strtod(p, &next);
      if (next == p) {
        ++p;
        continue;
      }
      p = next;
      // Expect y next.
      while (p < end && std::isspace(static_cast<unsigned char>(*p))) {
        ++p;
      }
      char* nextY = nullptr;
      double y = std::strtod(p, &nextY);
      if (nextY == p) {
        continue;
      }
      p = nextY;
      if (!any) {
        minX = maxX = v;
        minY = maxY = y;
        any = true;
      } else {
        minX = std::min(minX, v);
        maxX = std::max(maxX, v);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
      }
    }
    if (!any) {
      return std::nullopt;
    }
    return std::array<double, 4>{minX, minY, maxX, maxY};
  };

  if (auto c =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(expr)) {
    if (c->hasValueVector()) {
      auto const& vec = c->valueVector();
      if (vec && vec->size() > 0 && !vec->isNullAt(0)) {
        if (auto* sv = vec->asUnchecked<SimpleVector<StringView>>()) {
          auto s = sv->valueAt(0);
          return aabbFromBytes(std::string_view(s.data(), s.size()));
        }
      }
    } else {
      try {
        auto s = c->value().value<std::string>();
        return aabbFromBytes(s);
      } catch (...) {
        // Not a string Variant.
      }
    }
  }

  if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expr)) {
    if ((endsWithIgnoreCase(call->name(), "st_geometryfromtext") ||
         endsWithIgnoreCase(call->name(), "st_geomfromtext") ||
         endsWithIgnoreCase(call->name(), "st_geomfrombinary")) &&
        call->inputs().size() == 1) {
      return constantGeometryAabb(call->inputs()[0]);
    }
  }
  return std::nullopt;
}

/// Try to parse a single spatial call (Within / Intersects / Contains).
std::optional<SpatialEnvelopePrune> trySpatialEnvelopePruneFromCall(
    const core::TypedExprPtr& filter,
    const RowTypePtr& probeType,
    const RowTypePtr& buildType) {
  auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(filter);
  if (!call || call->inputs().size() != 2) {
    return std::nullopt;
  }

  SpatialPrunePredicate pred;
  if (endsWithIgnoreCase(call->name(), "st_within")) {
    pred = SpatialPrunePredicate::kWithin;
  } else if (endsWithIgnoreCase(call->name(), "st_intersects")) {
    pred = SpatialPrunePredicate::kIntersects;
  } else if (endsWithIgnoreCase(call->name(), "st_contains")) {
    pred = SpatialPrunePredicate::kWithin;
  } else {
    return std::nullopt;
  }

  auto leftRef = geometryFieldRef(call->inputs()[0]);
  auto rightRef = geometryFieldRef(call->inputs()[1]);

  // Intersects(constant, buildField) — constant AABB prefilter for build.
  if (pred == SpatialPrunePredicate::kIntersects) {
    auto leftAabb = constantGeometryAabb(call->inputs()[0]);
    auto rightAabb = constantGeometryAabb(call->inputs()[1]);
    if (leftAabb && rightRef && !leftRef) {
      if (buildType->containsChild(rightRef->first) &&
          !probeType->containsChild(rightRef->first)) {
        SpatialEnvelopePrune prune;
        prune.predicate = SpatialPrunePredicate::kIntersects;
        prune.buildGeometryName = rightRef->first;
        prune.buildIsWkb = rightRef->second;
        prune.buildConstantAabb = leftAabb;
        return prune;
      }
    }
    if (rightAabb && leftRef && !rightRef) {
      if (buildType->containsChild(leftRef->first) &&
          !probeType->containsChild(leftRef->first)) {
        SpatialEnvelopePrune prune;
        prune.predicate = SpatialPrunePredicate::kIntersects;
        prune.buildGeometryName = leftRef->first;
        prune.buildIsWkb = leftRef->second;
        prune.buildConstantAabb = rightAabb;
        return prune;
      }
    }
  }

  if (!leftRef || !rightRef) {
    return std::nullopt;
  }

  if (endsWithIgnoreCase(call->name(), "st_contains")) {
    std::swap(leftRef, rightRef);
  }

  const auto& leftField = leftRef->first;
  const auto& rightField = rightRef->first;
  const bool leftWkb = leftRef->second;
  const bool rightWkb = rightRef->second;

  const bool leftOnProbe = probeType->containsChild(leftField);
  const bool leftOnBuild = buildType->containsChild(leftField);
  const bool rightOnProbe = probeType->containsChild(rightField);
  const bool rightOnBuild = buildType->containsChild(rightField);

  SpatialEnvelopePrune prune;
  prune.predicate = pred;

  if (pred == SpatialPrunePredicate::kWithin) {
    if (leftOnProbe && rightOnBuild && !leftOnBuild && !rightOnProbe) {
      prune.probeGeometryName = leftField;
      prune.buildGeometryName = rightField;
      prune.withinPointOnBuild = false;
      prune.probeIsWkb = leftWkb;
      prune.buildIsWkb = rightWkb;
      return prune;
    }
    if (leftOnBuild && rightOnProbe && !leftOnProbe && !rightOnBuild) {
      prune.probeGeometryName = rightField;
      prune.buildGeometryName = leftField;
      prune.withinPointOnBuild = true;
      prune.probeIsWkb = rightWkb;
      prune.buildIsWkb = leftWkb;
      return prune;
    }
    return std::nullopt;
  }

  if (leftOnProbe && rightOnBuild && !leftOnBuild && !rightOnProbe) {
    prune.probeGeometryName = leftField;
    prune.buildGeometryName = rightField;
    prune.probeIsWkb = leftWkb;
    prune.buildIsWkb = rightWkb;
    return prune;
  }
  if (leftOnBuild && rightOnProbe && !leftOnProbe && !rightOnBuild) {
    prune.probeGeometryName = rightField;
    prune.buildGeometryName = leftField;
    prune.probeIsWkb = rightWkb;
    prune.buildIsWkb = leftWkb;
    return prune;
  }
  return std::nullopt;
}

} // namespace

namespace {
bool exprReferencesFieldName(
    const core::TypedExprPtr& expr,
    const std::string& name) {
  if (!expr) {
    return false;
  }
  if (auto field =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr)) {
    if (field->name() == name) {
      return true;
    }
  }
  for (auto const& input : expr->inputs()) {
    if (exprReferencesFieldName(input, name)) {
      return true;
    }
  }
  return false;
}
} // namespace

namespace {

std::optional<int64_t> constantIntegralValue(const core::TypedExprPtr& expr) {
  auto c = std::dynamic_pointer_cast<const core::ConstantTypedExpr>(expr);
  if (!c) {
    return std::nullopt;
  }
  try {
    switch (c->type()->kind()) {
      case TypeKind::BIGINT:
        return c->value().value<int64_t>();
      case TypeKind::INTEGER:
        return static_cast<int64_t>(c->value().value<int32_t>());
      case TypeKind::SMALLINT:
        return static_cast<int64_t>(c->value().value<int16_t>());
      case TypeKind::TINYINT:
        return static_cast<int64_t>(c->value().value<int8_t>());
      default:
        return std::nullopt;
    }
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<bool> constantBoolValue(const core::TypedExprPtr& expr) {
  auto c = std::dynamic_pointer_cast<const core::ConstantTypedExpr>(expr);
  if (!c || c->type()->kind() != TypeKind::BOOLEAN) {
    return std::nullopt;
  }
  try {
    return c->value().value<bool>();
  } catch (...) {
    return std::nullopt;
  }
}

/// Parse ST_KNN(probeGeom, buildGeom, k [, useSpheroid]).
std::optional<SpatialEnvelopePrune> trySpatialKnnFromFilter(
    const core::TypedExprPtr& filter,
    const RowTypePtr& probeType,
    const RowTypePtr& buildType) {
  auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(filter);
  if (!call || !endsWithIgnoreCase(call->name(), "st_knn")) {
    return std::nullopt;
  }
  if (call->inputs().size() != 3 && call->inputs().size() != 4) {
    return std::nullopt;
  }
  auto leftRef = geometryFieldRef(call->inputs()[0]);
  auto rightRef = geometryFieldRef(call->inputs()[1]);
  if (!leftRef || !rightRef) {
    return std::nullopt;
  }
  auto kOpt = constantIntegralValue(call->inputs()[2]);
  if (!kOpt || *kOpt < 1 || *kOpt > 32) {
    return std::nullopt;
  }
  bool useSpheroid = false;
  if (call->inputs().size() == 4) {
    auto sph = constantBoolValue(call->inputs()[3]);
    if (!sph) {
      return std::nullopt;
    }
    useSpheroid = *sph;
  }
  // Only Euclidean KNN is implemented on GPU.
  if (useSpheroid) {
    return std::nullopt;
  }

  const auto& leftField = leftRef->first;
  const auto& rightField = rightRef->first;
  bool leftOnProbe = probeType->containsChild(leftField) &&
      !buildType->containsChild(leftField);
  bool rightOnBuild = buildType->containsChild(rightField) &&
      !probeType->containsChild(rightField);
  bool leftOnBuild = buildType->containsChild(leftField) &&
      !probeType->containsChild(leftField);
  bool rightOnProbe = probeType->containsChild(rightField) &&
      !buildType->containsChild(rightField);

  SpatialEnvelopePrune prune;
  prune.predicate = SpatialPrunePredicate::kKnn;
  prune.knnK = static_cast<int32_t>(*kOpt);
  prune.knnUseSpheroid = useSpheroid;

  if (leftOnProbe && rightOnBuild) {
    prune.probeGeometryName = leftField;
    prune.buildGeometryName = rightField;
    prune.probeIsWkb = leftRef->second;
    prune.buildIsWkb = rightRef->second;
    return prune;
  }
  if (rightOnProbe && leftOnBuild) {
    // ST_KNN(build, probe, k) — treat first arg as query side per Sedona;
    // if sides are swapped in SQL, still bind by type membership.
    prune.probeGeometryName = rightField;
    prune.buildGeometryName = leftField;
    prune.probeIsWkb = rightRef->second;
    prune.buildIsWkb = leftRef->second;
    return prune;
  }
  // Name collision (both sides share projected name): assume (probe, build) order.
  if (probeType->containsChild(leftField) &&
      buildType->containsChild(rightField)) {
    prune.probeGeometryName = leftField;
    prune.buildGeometryName = rightField;
    prune.probeIsWkb = leftRef->second;
    prune.buildIsWkb = rightRef->second;
    return prune;
  }
  return std::nullopt;
}

} // namespace

std::optional<SpatialEnvelopePrune> trySpatialEnvelopePruneFromFilter(
    const core::TypedExprPtr& filter,
    const RowTypePtr& probeType,
    const RowTypePtr& buildType) {
  if (!filter) {
    return std::nullopt;
  }

  if (auto knn = trySpatialKnnFromFilter(filter, probeType, buildType)) {
    return knn;
  }

  // Single spatial call.
  if (auto single = trySpatialEnvelopePruneFromCall(filter, probeType, buildType)) {
    return single;
  }

  // Compound AND: prefer a Within(probe,build) conjunct for envelope prune;
  // also pick up a constant-bbox ∩ build conjunct when present (Q6).
  std::vector<core::TypedExprPtr> conjuncts;
  collectAndConjuncts(filter, conjuncts);
  if (conjuncts.size() < 2) {
    return std::nullopt;
  }

  std::optional<SpatialEnvelopePrune> withinPrune;
  std::optional<std::array<double, 4>> buildAabb;
  size_t spatialConjunctIndex = conjuncts.size();
  for (size_t i = 0; i < conjuncts.size(); ++i) {
    auto parsed =
        trySpatialEnvelopePruneFromCall(conjuncts[i], probeType, buildType);
    if (!parsed) {
      continue;
    }
    if (parsed->buildConstantAabb.has_value() &&
        parsed->probeGeometryName.empty()) {
      // Constant ∩ build-only conjunct.
      buildAabb = parsed->buildConstantAabb;
      continue;
    }
    if (parsed->predicate == SpatialPrunePredicate::kWithin && !withinPrune) {
      withinPrune = parsed;
      spatialConjunctIndex = i;
    } else if (!withinPrune) {
      withinPrune = parsed; // fall back to Intersects etc.
      spatialConjunctIndex = i;
    }
  }
  if (!withinPrune) {
    return std::nullopt;
  }
  withinPrune->filterIsCompound = true;
  if (buildAabb) {
    withinPrune->buildConstantAabb = buildAabb;
  }

  // Only a Within conjunct is evaluated exactly by the indexed kernel, so only
  // then is it safe to drop it from refinement. Intersects differs from Within
  // on geometry boundaries, and the kernel may reclassify one as the other.
  if (withinPrune->predicate == SpatialPrunePredicate::kWithin) {
    std::vector<core::TypedExprPtr> residuals;
    for (size_t i = 0; i < conjuncts.size(); ++i) {
      if (i != spatialConjunctIndex) {
        residuals.push_back(conjuncts[i]);
      }
    }
    if (!residuals.empty()) {
      auto combined = residuals[0];
      for (size_t i = 1; i < residuals.size(); ++i) {
        combined = std::make_shared<core::CallTypedExpr>(
            BOOLEAN(),
            std::vector<core::TypedExprPtr>{combined, residuals[i]},
            "and");
      }
      // A residual that reads the geometry itself would still force the wide
      // gather, so there is nothing to gain and correctness to lose.
      if (!exprReferencesFieldName(combined, withinPrune->probeGeometryName) &&
          !exprReferencesFieldName(combined, withinPrune->buildGeometryName)) {
        withinPrune->residualFilter = std::move(combined);
      }
    }
  }
  return withinPrune;
}

void CudfNestedLoopJoinBridge::setData(
    std::optional<CudfNestedLoopJoinBridge::build_data_type> data) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(!data_.has_value(), "Bridge already has data");
    data_ = std::move(data);
    promises = std::move(promises_); // Extract promises to fulfill outside lock
  }
  notify(std::move(promises)); // Wake up all blocked probe operators
}

// Returns build data if available, otherwise returns a future to wait on.
// Called by probe operators in isBlocked().
std::optional<CudfNestedLoopJoinBridge::build_data_type>
CudfNestedLoopJoinBridge::dataOrFuture(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(!cancelled_, "Getting data after the build side is aborted");
  if (data_.has_value()) {
    return data_;
  }
  // Data not ready yet, create a promise that will be fulfilled by setData()
  promises_.emplace_back("CudfNestedLoopJoinBridge::dataOrFuture");
  *future = promises_.back().getSemiFuture();
  return std::nullopt; // Probe will block on the future
}

void CudfNestedLoopJoinBridge::setBuildStream(
    rmm::cuda_stream_view buildStream) {
  std::lock_guard<std::mutex> l(mutex_);
  buildStream_ = buildStream;
}

std::optional<rmm::cuda_stream_view>
CudfNestedLoopJoinBridge::getBuildStream() {
  std::lock_guard<std::mutex> l(mutex_);
  return buildStream_;
}

CudfNestedLoopJoinBridge::spatial_index_type
CudfNestedLoopJoinBridge::spatialIndex(
    const std::function<std::unique_ptr<SpatialIndex>()>& factory) {
  // Held across the build so concurrent drivers wait for the first one rather
  // than each materialising their own copy. This serialises index construction,
  // which is the point: it is a one-off cost against a per-driver ~14GB
  // allocation that cannot fit.
  std::lock_guard<std::mutex> l(spatialIndexMutex_);
  if (!spatialIndex_) {
    spatialIndex_ = factory();
  }
  return spatialIndex_;
}

// ============================================================================
// Build Operator Implementation
// ============================================================================
// Accumulates all build-side input batches in GPU memory and transfers them
// to the bridge when all input is received.

CudfNestedLoopJoinBuild::CudfNestedLoopJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::NestedLoopJoinNode> joinNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          nullptr,
          joinNode->id(),
          "CudfNestedLoopJoinBuild",
          nvtx3::rgb{65, 105, 225}, // Royal Blue
          NvtxMethodFlag::kNoMoreInput,
          std::nullopt,
          joinNode),
      joinNode_(joinNode) {}

// Accumulates input batches in memory.
// All batches are kept as CudfVectors (GPU memory) until join completes.
void CudfNestedLoopJoinBuild::doAddInput(RowVectorPtr input) {
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    inputs_.push_back(std::move(cudfInput)); // Store in GPU memory
  }
}

bool CudfNestedLoopJoinBuild::needsInput() const {
  return !noMoreInput_;
}

RowVectorPtr CudfNestedLoopJoinBuild::doGetOutput() {
  return nullptr;
}

// Called when upstream finishes. Coordinates with peer build operators
// to transfer accumulated data to the bridge.
//
// Multi-driver coordination:
// - Multiple build operators may run in parallel (one per driver)
// - allPeersFinished() chooses ONE operator to collect and transfer data
// - Other operators just return and mark themselves finished
// - The chosen operator collects data from all peers and sets it on the bridge
void CudfNestedLoopJoinBuild::doNoMoreInput() {
  Operator::noMoreInput();

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;

  // Synchronization point: only the LAST driver to finish will proceed
  // Other drivers return here and will be woken when data transfer completes
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return; // Not the last driver - just wait
  }

  // This driver was chosen to collect data from all peers
  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    auto* build = dynamic_cast<CudfNestedLoopJoinBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    inputs_.insert(
        inputs_.end(),
        std::make_move_iterator(build->inputs_.begin()),
        std::make_move_iterator(build->inputs_.end()));
  }

  // Wake up peer build operators when we finish transferring data
  SCOPE_EXIT {
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue(); // Unblock other build operators
    }
  };

  // Concatenate all input batches into a single cuDF table.
  // getConcatenatedTable throws if the total row count exceeds cudf::size_type
  // limits (~2.1B rows). We don't use getConcatenatedTableBatched here because
  // batching the build side does not prevent output overflow for NLJ: a cross
  // join output is probe_rows × build_rows regardless of how the build is
  // split.
  auto stream = cudfGlobalStreamPool().get_stream();
  auto table = getConcatenatedTable(
      std::exchange(inputs_, {}),
      joinNode_->sources()[1]->outputType(),
      stream,
      get_output_mr());

  // Transfer build data to bridge - this will unblock probe operators.
  // No stream sync is required: the probe side uses syncBuildStream() via a
  // CUDA event to ensure the build table is ready before reading it.
  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto bridge = std::dynamic_pointer_cast<CudfNestedLoopJoinBridge>(joinBridge);

  bridge->setBuildStream(stream); // Pass stream for CUDA synchronization
  bridge->setData(
      std::make_optional(
          std::shared_ptr<cudf::table>(std::move(table)))); // Wake probes
}

exec::BlockingReason CudfNestedLoopJoinBuild::isBlocked(
    ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfNestedLoopJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

void CudfNestedLoopJoinBuild::doClose() {
  inputs_.clear();
  Operator::close();
}

// ============================================================================
// Probe Operator Implementation
// ============================================================================
// Performs the actual nested loop join by combining probe batches with
// build data using cuDF's cross_join or conditional_inner_join APIs.

CudfNestedLoopJoinProbe::CudfNestedLoopJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::NestedLoopJoinNode> joinNode,
    std::optional<SpatialEnvelopePrune> spatialPrune)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          joinNode->outputType(),
          joinNode->id(),
          "CudfNestedLoopJoinProbe",
          nvtx3::rgb{0, 128, 128}, // Teal
          NvtxMethodFlag::kGetOutput | NvtxMethodFlag::kNoMoreInput,
          std::nullopt,
          joinNode),
      joinNode_(joinNode),
      spatialPrune_(std::move(spatialPrune)) {
  joinType_ = joinNode_->joinType();
  probeType_ = joinNode_->sources()[0]->outputType();
  buildType_ = joinNode_->sources()[1]->outputType();

  // For kLeftSemiProject, the last output column is a BOOLEAN match flag
  // that doesn't exist in probe or build types — skip it during resolution.
  auto numColumnsToResolve = outputType_->size();
  if (joinType_ == core::JoinType::kLeftSemiProject) {
    VELOX_CHECK_GE(numColumnsToResolve, 1);
    --numColumnsToResolve;
  }

  for (size_t i = 0; i < numColumnsToResolve; ++i) {
    const auto& name = outputType_->nameOf(i);
    auto probeIdx = probeType_->getChildIdxIfExists(name);
    if (probeIdx.has_value()) {
      probeColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(probeIdx.value()));
      probeColumnOutputIndices_.push_back(i);
      continue;
    }
    auto buildIdx = buildType_->getChildIdxIfExists(name);
    if (buildIdx.has_value()) {
      buildColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(buildIdx.value()));
      buildColumnOutputIndices_.push_back(i);
      continue;
    }
    VELOX_FAIL("Output column not found in probe or build types: {}", name);
  }

  if (spatialPrune_.has_value()) {
    probeGeomChannel_ = static_cast<cudf::size_type>(
        probeType_->getChildIdx(spatialPrune_->probeGeometryName));
    buildGeomChannel_ = static_cast<cudf::size_type>(
        buildType_->getChildIdx(spatialPrune_->buildGeometryName));
    if (spatialPrune_->buildRadiusName.has_value()) {
      buildRadiusChannel_ = static_cast<cudf::size_type>(
          buildType_->getChildIdx(spatialPrune_->buildRadiusName.value()));
    }
  }
}

void CudfNestedLoopJoinProbe::initialize() {
  // Filter construction is deferred from the ctor to avoid memory allocation
  // during driver initialization. Mirrors #17045 for CudfHashJoinProbe.
  Operator::initialize();

  if (!joinNode_->joinCondition()) {
    return;
  }

  exec::ExprSet exprs({joinNode_->joinCondition()}, operatorCtx_->execCtx());
  VELOX_CHECK_EQ(exprs.exprs().size(), 1);

  // Resolve the session timezone once so timezone-sensitive CudfFunctions built
  // on the precompute path receive it at construction.
  const auto context =
      contextFromConfig(operatorCtx_->driverCtx()->queryConfig());

  // Non-AST sub-expressions that span both sides (e.g. ST_Distance(probe, build)
  // <= radius) cannot be precomputed on either side alone. Evaluate the whole
  // condition generally against the cross product instead of building an AST.
  if (hasNonAstSubexprSpanningBothSides(
          exprs.exprs()[0], probeType_, buildType_)) {
    useAstFilter_ = false;
    filterEvaluator_ = createCudfExpression(
        exprs.exprs()[0],
        facebook::velox::type::concatRowTypes({probeType_, buildType_}),
        context);
    hasFilter_ = true;
    if (spatialPrune_.has_value() && spatialPrune_->residualFilter) {
      exec::ExprSet residualExprs(
          {spatialPrune_->residualFilter}, operatorCtx_->execCtx());
      VELOX_CHECK_EQ(residualExprs.exprs().size(), 1);
      auto const combinedType =
          facebook::velox::type::concatRowTypes({probeType_, buildType_});
      residualEvaluator_ = createCudfExpression(
          residualExprs.exprs()[0], combinedType, context);
      residualUnreadColumn_.assign(combinedType->size(), false);
      for (size_t i = 0; i < combinedType->size(); ++i) {
        residualUnreadColumn_[i] = !exprReferencesFieldName(
            spatialPrune_->residualFilter, combinedType->nameOf(i));
      }
    }
    return;
  }

  // Convert Velox expression to cuDF AST expression tree.
  // The AST will be passed to cudf::conditional_inner_join() for GPU
  // evaluation.
  createAstTree(
      exprs.exprs()[0],
      tree_,
      scalars_,
      probeType_,
      buildType_,
      leftPrecomputeInstructions_,
      rightPrecomputeInstructions_,
      context);

  // Set hasFilter_ only after the AST has been fully built so that a throw
  // from createAstTree() does not leave the operator marked as having a filter
  // with a partially-initialized tree.
  hasFilter_ = true;
}

void CudfNestedLoopJoinProbe::doClose() {
  Operator::close();
  buildData_.reset();
  probeMatchedFlags_.reset();
  buildMatchedFlags_.reset();
  buildPrecomputed_.clear();
  scalars_.clear();
  tree_ = {};
  filterEvaluator_.reset();
}

bool CudfNestedLoopJoinProbe::needsInput() const {
  return !noMoreInput_ && !finished_ && input_ == nullptr &&
      buildData_.has_value();
}

void CudfNestedLoopJoinProbe::doAddInput(RowVectorPtr input) {
  // Skip input processing when build is empty for join types with no output.
  if (skipInput_) {
    VELOX_CHECK_NULL(input_);
    return;
  }
  VELOX_CHECK_NULL(input_, "Probe input already set");
  input_ = std::move(input);
  probeMatchedFlags_.reset();
}

void CudfNestedLoopJoinProbe::doNoMoreInput() {
  Operator::noMoreInput();

  if (!isRightOrFullJoin()) {
    return;
  }

  // Empty build has no matched flags to merge across peers.
  if (buildEmpty_) {
    return;
  }

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;

  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(),
          operatorCtx_->driver(),
          &peerFuture_,
          promises,
          peers)) {
    return;
  }

  SCOPE_EXIT {
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  isLastDriver_ = true;

  // Unfiltered cross_join matches every build row on every probe batch, so
  // every driver's buildMatchedFlags_ would be all-true. Skip the stream-join
  // and peer merge when there is no filter.
  if (!buildEmpty_ && hasFilter_) {
    auto stream = cudfGlobalStreamPool().get_stream();

    // GPU stream synchronization: allPeersFinished synchronizes CPU threads
    // but not GPU streams. A peer's CPU thread may have returned from
    // getOutput() while its GPU work (updating buildMatchedFlags_) is still
    // in flight. join_streams establishes GPU-side ordering.
    std::vector<rmm::cuda_stream_view> inputStreams;
    if (lastProbeStream_.has_value()) {
      inputStreams.push_back(lastProbeStream_.value());
    }
    for (auto& peer : peers) {
      if (peer.get() == operatorCtx_->driver()) {
        continue;
      }
      auto op = peer->findOperator(planNodeId());
      auto* probe = dynamic_cast<CudfNestedLoopJoinProbe*>(op);
      if (probe != nullptr && probe->lastProbeStream_.has_value()) {
        inputStreams.push_back(probe->lastProbeStream_.value());
      }
    }
    if (!inputStreams.empty()) {
      cudf::detail::join_streams(inputStreams, stream);
    }

    // Merge buildMatchedFlags_ from all peers via BITWISE_OR.
    for (auto& peer : peers) {
      if (peer.get() == operatorCtx_->driver()) {
        continue;
      }
      auto op = peer->findOperator(planNodeId());
      auto* probe = dynamic_cast<CudfNestedLoopJoinProbe*>(op);
      if (probe == nullptr) {
        continue;
      }
      auto orResult = cudf::binary_operation(
          buildMatchedFlags_->view(),
          probe->buildMatchedFlags_->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          get_temp_mr());
      // binary_operation is async on `stream`; the old column destructs via
      // cudaFreeAsync on its allocation stream (not `stream`), so the free
      // can race the kernel. Drain `stream` before the move-assign.
      stream.synchronize();
      buildMatchedFlags_ = std::move(orResult);
    }
  }
}

bool CudfNestedLoopJoinProbe::isFinished() {
  if (finished_) {
    return true;
  }
  // For right/full join, the last driver must not finish until build mismatch
  // rows have been emitted. Non-last drivers finish normally.
  if (isRightOrFullJoin() && noMoreInput_ && input_ == nullptr) {
    if (!isLastDriver_) {
      return true;
    }
    return buildMismatchEmitted_;
  }
  return false;
}

exec::BlockingReason CudfNestedLoopJoinProbe::isBlocked(
    ContinueFuture* future) {
  // For right/full join: after build data is available, also block on peer
  // probes finishing (allPeersFinished barrier in noMoreInput).
  if (isRightOrFullJoin() && buildData_.has_value()) {
    if (!peerFuture_.valid()) {
      return exec::BlockingReason::kNotBlocked;
    }
    *future = std::move(peerFuture_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }

  if (buildData_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto bridge = std::dynamic_pointer_cast<CudfNestedLoopJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(bridge);
  VELOX_CHECK_NOT_NULL(future);

  buildData_ = bridge->dataOrFuture(future);
  if (!buildData_.has_value()) {
    return exec::BlockingReason::kWaitForJoinBuild;
  }

  buildStream_ = bridge->getBuildStream();

  if (buildData_.value()->num_rows() == 0) {
    buildEmpty_ = true;
    // For inner/right join, set skipInput_ to consume probe batches without
    // processing (prevents upstream exchange hanging). Match CPU NLJ behavior
    // which always consumes input rather than finishing early.
    if (skipProbeOnEmptyBuild()) {
      skipInput_ = true;
    }
  }

  // Initialize build matched flags for filtered right/full join (single BOOL8
  // column with one element per build row, all false). Unfiltered cross_join
  // matches every build row, so flags aren't needed.
  if (isRightOrFullJoin() && hasFilter_ && !buildEmpty_) {
    auto initStream = cudfGlobalStreamPool().get_stream();
    auto numRows = buildData_.value()->num_rows();
    auto falseScalar =
        cudf::numeric_scalar<bool>(false, true, initStream, get_temp_mr());
    buildMatchedFlags_ = cudf::make_column_from_scalar(
        falseScalar, numRows, initStream, get_temp_mr());
    initStream.synchronize();
  }

  // Precompute build-side sub-expressions for filter evaluation (once, here,
  // since the build table is fixed for the lifetime of this probe operator).
  if (hasFilter_ && !rightPrecomputeInstructions_.empty() && !buildEmpty_) {
    auto precomputeStream = cudfGlobalStreamPool().get_stream();
    auto buildColumnViews = tableViewToColumnViews(buildData_.value()->view());
    buildPrecomputed_ = precomputeSubexpressions(
        buildColumnViews,
        rightPrecomputeInstructions_,
        scalars_,
        buildType_,
        precomputeStream);
    buildExtendedView_ =
        createExtendedTableView(buildData_.value()->view(), buildPrecomputed_);
    precomputeStream.synchronize();
  }

  if (spatialPrune_.has_value() && !buildEmpty_) {
    auto indexStream = cudfGlobalStreamPool().get_stream();
    ensureSpatialIndex(indexStream);
  }

  return exec::BlockingReason::kNotBlocked;
}

void CudfNestedLoopJoinProbe::syncBuildStream(
    rmm::cuda_stream_view probeStream) {
  if (buildStream_.has_value()) {
    if (!cudaEvent_) {
      cudaEvent_ = std::make_unique<CudaEvent>();
    }
    cudaEvent_->recordFrom(buildStream_.value()).waitOn(probeStream);
    buildStream_.reset();
  }
}

namespace {

void throwIfInvalidGeometryType(
    rmm::device_scalar<int32_t>& flag,
    rmm::cuda_stream_view stream) {
  if (flag.value(stream) != 0) {
    VELOX_USER_FAIL("Invalid geometry input to spatial join envelope prune");
  }
}

// Average bytes one gathered output row costs across the emitted columns.
// String columns dominate: a Velox zone geometry blob runs ~1KB, so an output
// slice budgeted purely by row count asks for tens of GB in a single gather.
int64_t averageRowBytes(
    cudf::table_view const& view,
    rmm::cuda_stream_view stream) {
  int64_t bytes = 0;
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    auto const& col = view.column(i);
    if (col.type().id() == cudf::type_id::STRING) {
      cudf::strings_column_view scv(col);
      bytes += static_cast<int64_t>(sizeof(cudf::size_type));
      if (col.size() > 0) {
        bytes += static_cast<int64_t>(scv.chars_size(stream)) / col.size();
      }
    } else if (cudf::is_fixed_width(col.type())) {
      bytes += static_cast<int64_t>(cudf::size_of(col.type()));
    } else {
      // Nested/unsupported width: assume a pointer-ish cost rather than fail.
      bytes += 8;
    }
  }
  return bytes;
}

} // namespace

void CudfNestedLoopJoinProbe::ensureSpatialIndex(rmm::cuda_stream_view stream) {
  if (spatialIndex_ || !spatialPrune_.has_value() || !buildData_.has_value()) {
    return;
  }
  VELOX_NVTX_FUNC_RANGE();
  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto bridge = std::dynamic_pointer_cast<CudfNestedLoopJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(bridge);
  // Built once per worker and shared with the other probe drivers: the
  // converted build geometry alone is ~14GB for SF100 zones.
  spatialIndex_ = bridge->spatialIndex([&]() {
    return buildSpatialIndex(stream);
  });
}

std::unique_ptr<CudfNestedLoopJoinBridge::SpatialIndex>
CudfNestedLoopJoinProbe::buildSpatialIndex(rmm::cuda_stream_view stream) {
  AllocLabelGuard allocLabel("spatial.buildIndex");
  auto index = std::make_unique<CudfNestedLoopJoinBridge::SpatialIndex>();
  auto mr = get_temp_mr();
  auto buildView = buildData_.value()->view();
  cudf::column_view geom = buildView.column(buildGeomChannel_);
  {
    rmm::device_scalar<int32_t> wkbInvalid(0, stream, mr);
    index->buildGeomVelox = ensureVeloxGeometryColumn(
        geom,
        spatialPrune_->buildIsWkb,
        wkbInvalid.data(),
        stream,
        mr);
    if (index->buildGeomVelox) {
      geom = index->buildGeomVelox->view();
    }
  }
  std::unique_ptr<cudf::column> emptyExpandCol;
  cudf::column_view expandBy;
  if (buildRadiusChannel_.has_value()) {
    expandBy = buildView.column(buildRadiusChannel_.value());
  } else {
    emptyExpandCol = cudf::make_empty_column(cudf::type_id::FLOAT64);
    expandBy = emptyExpandCol->view();
  }

  rmm::device_scalar<int32_t> invalid(0, stream, mr);
  auto envelopes = extractGeometryEnvelopes(
      geom, expandBy, /*constantExpandBy=*/0.0, invalid.data(), stream, mr);

  // Q6: drop build envelopes that miss the constant bbox so the grid never
  // indexes zones outside ST_Intersects(constant_poly, zone).
  if (spatialPrune_->buildConstantAabb.has_value()) {
    auto const& aabb = *spatialPrune_->buildConstantAabb;
    auto mkScalar = [&](double v) {
      return cudf::numeric_scalar<double>(v, true, stream, mr);
    };
    auto aabbMinX = mkScalar(aabb[0]);
    auto aabbMinY = mkScalar(aabb[1]);
    auto aabbMaxX = mkScalar(aabb[2]);
    auto aabbMaxY = mkScalar(aabb[3]);
    // intersects = !(maxX < aabbMinX || minX > aabbMaxX ||
    //                maxY < aabbMinY || minY > aabbMaxY)
    auto notLeft = cudf::binary_operation(
        envelopes.maxX->view(),
        aabbMinX,
        cudf::binary_operator::GREATER_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto notRight = cudf::binary_operation(
        envelopes.minX->view(),
        aabbMaxX,
        cudf::binary_operator::LESS_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto notBelow = cudf::binary_operation(
        envelopes.maxY->view(),
        aabbMinY,
        cudf::binary_operator::GREATER_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto notAbove = cudf::binary_operation(
        envelopes.minY->view(),
        aabbMaxY,
        cudf::binary_operator::LESS_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto and1 = cudf::binary_operation(
        notLeft->view(),
        notRight->view(),
        cudf::binary_operator::LOGICAL_AND,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto and2 = cudf::binary_operation(
        notBelow->view(),
        notAbove->view(),
        cudf::binary_operator::LOGICAL_AND,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto intersects = cudf::binary_operation(
        and1->view(),
        and2->view(),
        cudf::binary_operator::LOGICAL_AND,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    // Null out non-intersecting envelopes (grid skips null build rows).
    // copy_if_else(lhs, rhs, mask): mask true → lhs, false → rhs.
    auto nullDouble =
        cudf::numeric_scalar<double>(0.0, false /*is_valid*/, stream, mr);
    auto nullify = [&](std::unique_ptr<cudf::column>& col) {
      col = cudf::copy_if_else(
          col->view(), nullDouble, intersects->view(), stream, mr);
    };
    nullify(envelopes.minX);
    nullify(envelopes.minY);
    nullify(envelopes.maxX);
    nullify(envelopes.maxY);
  }

  // Null envelopes for unsupported rows; continue indexing the rest.
  index->envelopeGrid =
      buildGeometryEnvelopeGrid(std::move(envelopes), stream, mr);

  // When the build side is the polygon side (Q11: zones broadcast on build),
  // also build a per-ring envelope decomposition. Antimeridian/island
  // multipolygons have globe-spanning whole-geometry bboxes; per-ring bboxes
  // are tight, so the point-batch grid query rejects candidates that fall in
  // the coarse bbox but nowhere near the real polygon.
  index->buildIsPolygonIndex = !geometryColumnLooksLikePoints(geom, stream);
  if (index->buildIsPolygonIndex) {
    constexpr int32_t kMaxPartsPerRow = 65536;
    rmm::device_scalar<int32_t> partInvalid(0, stream, mr);
    auto parts = extractGeometryPartEnvelopes(
        geom,
        expandBy,
        /*constantExpandBy=*/0.0,
        kMaxPartsPerRow,
        partInvalid.data(),
        stream,
        mr);
    index->partEnvelopes = std::move(parts.envelopes);
    index->partToRow = std::move(parts.partToRow);
  }
  // Other drivers read this index from their own streams, so make the
  // construction visible before publishing it.
  stream.synchronize();
  return index;
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
CudfNestedLoopJoinProbe::spatialPruneConditionalIndices(
    cudf::table_view probeTableView,
    cudf::table_view buildView,
    rmm::cuda_stream_view stream,
    bool needBuildIndices,
    cudf::size_type probeRowBegin,
    cudf::size_type* probeRowsConsumed) {
  VELOX_NVTX_FUNC_RANGE();
  auto mr = get_temp_mr();
  ensureSpatialIndex(stream);
  VELOX_CHECK_NOT_NULL(spatialIndex_);
  VELOX_CHECK_NOT_NULL(
      filterEvaluator_,
      "Join filter evaluator must be initialized before "
      "spatialPruneConditionalIndices");

  const auto numProbeRows = probeTableView.num_rows();
  const auto numBuildRows = buildView.num_rows();
  if (numProbeRows == 0 || numBuildRows == 0) {
    return {
        cudf::make_empty_column(cudf::type_to_id<cudf::size_type>()),
        needBuildIndices
            ? cudf::make_empty_column(cudf::type_to_id<cudf::size_type>())
            : nullptr};
  }

  // Any allocation inside this function that is not covered by a tighter label
  // below reports this one, which distinguishes "here but untagged" from
  // "somewhere else entirely".
  AllocLabelGuard functionAllocLabel("spatial.untagged");

  auto emptyExpandCol = cudf::make_empty_column(cudf::type_id::FLOAT64);
  rmm::device_scalar<int32_t> invalid(0, stream, mr);
  cudf::column_view probeGeomCol = probeTableView.column(probeGeomChannel_);
  std::unique_ptr<cudf::column> probeGeomVelox;
  {
    AllocLabelGuard allocLabel("spatial.probeGeomConvert");
    probeGeomVelox = ensureVeloxGeometryColumn(
        probeGeomCol,
        spatialPrune_->probeIsWkb,
        invalid.data(),
        stream,
        mr);
  }
  if (probeGeomVelox) {
    probeGeomCol = probeGeomVelox->view();
  }

  cudf::column_view buildGeomCol = buildView.column(buildGeomChannel_);
  if (spatialIndex_->buildGeomVelox) {
    buildGeomCol = spatialIndex_->buildGeomVelox->view();
  }

  // ST_KNN: expanding envelope-grid search + exact top-k distances.
  // Never materializes probe×build cross product.
  if (spatialPrune_->predicate == SpatialPrunePredicate::kKnn) {
    AllocLabelGuard allocLabel("spatial.knn");
    VELOX_CHECK(
        !spatialPrune_->knnUseSpheroid,
        "ST_KNN spheroid distance is not supported on GPU");
    const int32_t knnK = spatialPrune_->knnK;
    VELOX_CHECK_GT(knnK, 0);
    VELOX_CHECK_LE(knnK, 32);

    // Bound work per getOutput so the stuck-operator watchdog (~30 min) is
    // never tripped. SF100 trips arrive in large probe inputs; walking tens of
    // thousands of probes through expanding-radius KNN in one call hung for
    // 40+ minutes. Yield via probeRowsConsumed after a few small batches.
    constexpr cudf::size_type kMaxKnnProbeBatchCap = 256;
    constexpr int64_t kMaxCandBudget = 32'000'000;
    constexpr int64_t kMaxKnnMatchRowsPerGetOutput = 8'000; // ~1.6k probes at k=5
    constexpr int32_t kMaxKnnBatchesPerGetOutput = 4;
    constexpr auto kMaxKnnGetOutputWall = std::chrono::seconds(15);
    const int64_t numBuildRows = buildGeomCol.size();
    const cudf::size_type kMaxKnnProbeBatch = numBuildRows <= 0
        ? kMaxKnnProbeBatchCap
        : static_cast<cudf::size_type>(std::clamp<int64_t>(
              kMaxCandBudget / numBuildRows, 16, kMaxKnnProbeBatchCap));
    const int64_t outputRowBytes =
        averageRowBytes(
            probeTableView.select(probeColumnIndicesToGather_), stream) +
        averageRowBytes(buildView.select(buildColumnIndicesToGather_), stream);
    // Memory budget is secondary; the stuck-operator yield caps are primary.
    const int64_t memoryRowBudget = outputRowBytes <= 0
        ? kMaxKnnMatchRowsPerGetOutput
        : std::clamp<int64_t>(
              (int64_t{256} << 20) / std::max<int64_t>(outputRowBytes, 1),
              1'000,
              kMaxKnnMatchRowsPerGetOutput);
    const int64_t outputRowBudget =
        std::min<int64_t>(memoryRowBudget, kMaxKnnMatchRowsPerGetOutput);

    std::vector<std::unique_ptr<cudf::column>> matchedProbeChunks;
    std::vector<std::unique_ptr<cudf::column>> matchedBuildChunks;
    int64_t accumulatedMatches = 0;
    cudf::size_type consumedRows = numProbeRows;
    rmm::device_scalar<int32_t> knnInvalid(0, stream, mr);
    int32_t batchesDone = 0;
    const auto knnDeadline =
        std::chrono::steady_clock::now() + kMaxKnnGetOutputWall;

    for (cudf::size_type begin = probeRowBegin; begin < numProbeRows;) {
      if (accumulatedMatches >= outputRowBudget) {
        consumedRows = begin;
        break;
      }
      if (batchesDone >= kMaxKnnBatchesPerGetOutput) {
        consumedRows = begin;
        break;
      }
      if (batchesDone > 0 &&
          std::chrono::steady_clock::now() >= knnDeadline) {
        consumedRows = begin;
        break;
      }
      const cudf::size_type batch = std::min<cudf::size_type>(
          kMaxKnnProbeBatch, numProbeRows - begin);
      auto probeSlice =
          cudf::slice(probeGeomCol, {begin, begin + batch}, stream)[0];
      auto [batchProbe, batchBuild] = geometryKnnJoinIndices(
          probeSlice,
          buildGeomCol,
          spatialIndex_->envelopeGrid,
          knnK,
          knnInvalid.data(),
          stream,
          mr);
      // Shift probe indices from batch-local to input-local.
      // Must use cudf::binary_operation: this .cpp is compiled by host g++,
      // so thrust device lambdas are not allowed.
      if (batchProbe->size() > 0 && begin > 0) {
        cudf::numeric_scalar<cudf::size_type> base(begin, true, stream, mr);
        batchProbe = cudf::binary_operation(
            batchProbe->view(),
            base,
            cudf::binary_operator::ADD,
            cudf::data_type{cudf::type_to_id<cudf::size_type>()},
            stream,
            mr);
      }
      accumulatedMatches += batchProbe->size();
      matchedProbeChunks.push_back(std::move(batchProbe));
      if (needBuildIndices) {
        matchedBuildChunks.push_back(std::move(batchBuild));
      }
      begin += batch;
      consumedRows = begin;
      ++batchesDone;
      if (accumulatedMatches >= outputRowBudget && begin < numProbeRows) {
        break;
      }
    }

    if (probeRowsConsumed) {
      *probeRowsConsumed = consumedRows;
    }
    if (matchedProbeChunks.empty()) {
      return {
          cudf::make_empty_column(cudf::type_to_id<cudf::size_type>()),
          needBuildIndices
              ? cudf::make_empty_column(cudf::type_to_id<cudf::size_type>())
              : nullptr};
    }
    std::vector<cudf::column_view> probeViews;
    for (auto& c : matchedProbeChunks) {
      probeViews.push_back(c->view());
    }
    auto concatProbe = cudf::concatenate(probeViews, stream, mr);
    std::unique_ptr<cudf::column> concatBuild;
    if (needBuildIndices) {
      std::vector<cudf::column_view> buildViews;
      for (auto& c : matchedBuildChunks) {
        buildViews.push_back(c->view());
      }
      concatBuild = cudf::concatenate(buildViews, stream, mr);
    }
    return {std::move(concatProbe), std::move(concatBuild)};
  }

  // Resolve Within point-side from geometry tags. FieldAccess names often
  // collide after ST_GeomFromBinary projection ("st_geomfrombinary" on both
  // sides), so withinPointOnBuild from filter parsing can be wrong.
  // Also: compound AND filters may have been mislabeled as Intersects before
  // AND-flattening; if tags show point vs poly, force Within.
  {
    bool const probeIsPoint =
        geometryColumnLooksLikePoints(probeGeomCol, stream);
    bool const buildIsPoint =
        geometryColumnLooksLikePoints(buildGeomCol, stream);
    if (probeIsPoint != buildIsPoint) {
      if (spatialPrune_->predicate == SpatialPrunePredicate::kIntersects) {
        // Point–polygon Intersects is Within for envelope/PIP purposes.
        spatialPrune_->predicate = SpatialPrunePredicate::kWithin;
      }
      if (spatialPrune_->predicate == SpatialPrunePredicate::kWithin) {
        spatialPrune_->withinPointOnBuild = buildIsPoint;
      }
    }
  }

  auto probeEnvelopes = [&] {
    AllocLabelGuard allocLabel("spatial.probeEnvelopes");
    return extractGeometryEnvelopes(
        probeGeomCol,
        emptyExpandCol->view(),
        /*constantExpandBy=*/0.0,
        invalid.data(),
        stream,
        mr);
  }();
  // Unsupported probe rows get null envelopes and are skipped by the grid.

  // Evaluate spatial predicate only on geometry columns for candidates.
  // Batch probe envelopes to bound candidate-index memory (cudf::size_type is
  // int32). Polygon-probe Within (points on build) always uses indexed
  // point-in-polygon (no multipolygon gather): Q4 has a small point build,
  // Q10 (zone ⟕ ~600M trips) has a large one. Both batch many zone polygons —
  // batch size 1 would serialize ~1M zones into as many GPU launches.
  constexpr int64_t kMaxCandidateRows = 8'000'000;
  constexpr int64_t kMaxPointProbeCandidateRows = 2'000'000;
  constexpr cudf::size_type kSmallPointBuildRows = 16'384;
  // Q6: ~tens of build zones after bbox filter × hundreds of millions of
  // probe points. Prefer indexing the small polygon build once and streaming
  // large point batches (inverted per-batch point grids thrash the GPU).
  constexpr cudf::size_type kSmallPolyBuildRows = 4'096;
  const bool polyProbeWithin =
      spatialPrune_->predicate == SpatialPrunePredicate::kWithin &&
      spatialPrune_->withinPointOnBuild;
  const bool pointProbeWithin =
      spatialPrune_->predicate == SpatialPrunePredicate::kWithin &&
      !spatialPrune_->withinPointOnBuild;
  const bool polyProbeSmallPointBuild =
      polyProbeWithin && numBuildRows <= kSmallPointBuildRows;
  const bool pointProbeSmallPolyBuild =
      pointProbeWithin && numBuildRows <= kSmallPolyBuildRows;
  // Invert point-probe Within only when the polygon build is large (Q11).
  const bool invertPointProbeWithin =
      pointProbeWithin && !pointProbeSmallPolyBuild;
  // Q10 (large point build): batch zone polygons so ~1M zones become a few
  // hundred grid queries instead of a per-zone serial loop. With the fine
  // 4096² point grid (+ skip-unique), candidates per batch stay manageable
  // even at larger probe batches — each pickup still falls in ~1 zone.
  // Invert path (Q11): the point grid is rebuilt per batch and every batch then
  // scans the full ~1M-part zone envelope set in queryGeometryEnvelopeGrid, so
  // the zone-scan cost is paid once per batch. A tiny 32K batch splits each
  // ~1.9M-row probe vector into ~58 passes, re-scanning all zones ~58× and
  // leaving the point grid coarse (side = sqrt(batch) ≈ 181 → poor selectivity).
  // Processing the whole vector in one batch collapses that to a single zone
  // scan and a fine grid (side up to 4096). Candidates stay tiny (~1-2 zones per
  // point ⇒ ~few M ≪ the 24M sub-batch cap), and the adaptive backoff below
  // still halves the batch if a pathological vector ever overflows.
  cudf::size_type kMaxProbeBatch = polyProbeSmallPointBuild ? 8'192
      : (polyProbeWithin ? 16'384
      : (pointProbeSmallPolyBuild ? 65'536
      : (invertPointProbeWithin ? 2'097'152 : 16'384)));
  if (!polyProbeWithin && !pointProbeWithin &&
      spatialIndex_->envelopeGrid.largeBuildIndices) {
    const auto numLarge =
        spatialIndex_->envelopeGrid.largeBuildIndices->size();
    if (numLarge > 0) {
      // Keep probeBatch * numLarge well under int32 match-index capacity.
      kMaxProbeBatch = std::min(
          kMaxProbeBatch,
          std::max<cudf::size_type>(
              16, static_cast<cudf::size_type>(2'000'000 / numLarge)));
    }
  }
  const int64_t candidateChunkRows =
      (invertPointProbeWithin || pointProbeSmallPolyBuild ||
       polyProbeSmallPointBuild)
      ? kMaxPointProbeCandidateRows
      : kMaxCandidateRows;
  std::vector<std::unique_ptr<cudf::column>> matchedProbeChunks;
  std::vector<std::unique_ptr<cudf::column>> matchedBuildChunks;

  // Candidates per batch are data dependent and highly skewed: most zone
  // envelopes cover a handful of points, but a few continent-scale ones each
  // sweep in hundreds of millions and overflow the int32 match counter. Tune
  // the batch to the data — back off on overflow, recover gradually — so the
  // common case still runs in large batches. A single probe row can never
  // overflow (its matches are bounded by the build row count), so batch size
  // 1 always makes progress.
  cudf::size_type batchCeiling = kMaxProbeBatch;
  cudf::size_type currentBatch = kMaxProbeBatch;
  int successStreak = 0;

  // Streaming budget: nested zones ⨝ 600M trips (Q10 SF100) yield billions of
  // matched pairs — more than one cudf column (2^31) or one output batch can
  // hold. Emit at most outputRowBudget matched rows per call, remembering how
  // far into the probe we got via probeRowsConsumed, and keep any single
  // sub-batch's candidate set bounded so one flush stays well under 2^31.
  //
  // The slice is bounded in bytes, not rows: the caller gathers these indices
  // into the emitted columns, and when one of those is a geometry blob (~1KB per
  // zone row) a flat 16M-row slice turns into a ~15GB single allocation that no
  // practical pool can satisfy. Row-count caps are kept as outer bounds so
  // narrow outputs still stream in large slices.
  constexpr int64_t kOutputByteBudget = int64_t{2} << 30; // 2 GiB per slice
  constexpr int64_t kMaxOutputRowBudget = 16'000'000;
  constexpr int64_t kMinOutputRowBudget = 256'000;
  const int64_t outputRowBytes =
      averageRowBytes(
          probeTableView.select(probeColumnIndicesToGather_), stream) +
      averageRowBytes(buildView.select(buildColumnIndicesToGather_), stream);
  const int64_t outputRowBudget = outputRowBytes <= 0
      ? kMaxOutputRowBudget // COUNT(*): no columns gathered
      : std::clamp<int64_t>(
            kOutputByteBudget / outputRowBytes,
            kMinOutputRowBudget,
            kMaxOutputRowBudget);
  constexpr int64_t kSubBatchCandidateCap = 24'000'000;
  LOG_FIRST_N(WARNING, 20) << "[spatialjoin] probeRows=" << numProbeRows
                           << " buildRows=" << buildView.num_rows()
                           << " outCols=" << probeColumnIndicesToGather_.size()
                           << "+" << buildColumnIndicesToGather_.size()
                           << " rowBytes=" << outputRowBytes
                           << " rowBudget=" << outputRowBudget;
  // Hoisted out of the candidate loop below: averageRowBytes synchronizes the
  // stream to read string sizes, and both table widths are loop-invariant.
  const int64_t refineRowBytes =
      (spatialPrune_->filterIsCompound && residualEvaluator_ == nullptr)
      ? std::max<int64_t>(
            1,
            averageRowBytes(probeTableView, stream) +
                averageRowBytes(buildView, stream))
      : 1;
  LOG_FIRST_N(WARNING, 10) << "[geo:path] pred="
                           << static_cast<int>(spatialPrune_->predicate)
                           << " pointOnBuild=" << spatialPrune_->withinPointOnBuild
                           << " compound=" << spatialPrune_->filterIsCompound
                           << " invert=" << invertPointProbeWithin
                           << " chunkRows=" << candidateChunkRows
                           << " refineRowBytes=" << refineRowBytes
                           << " probeCols=" << probeTableView.num_columns()
                           << " buildCols=" << buildView.num_columns();
  int64_t accumulatedMatches = 0;
  cudf::size_type consumedRows = numProbeRows;

  // Per-ring probe decomposition for the polygon-probe paths (Q4/Q10, where the
  // probe is the zone side). One antimeridian/island multipolygon otherwise
  // sweeps in ~20% of all trips as candidates; per-ring bboxes are tight so the
  // build point grid rejects them up front. Probe batching stays in row units
  // and slices the contiguous part range for each row batch, so a zone's rings
  // never straddle a batch boundary and per-batch dedup is exact.
  const bool probeUseParts = polyProbeWithin;
  GeometryEnvelopes probePartEnv;
  std::unique_ptr<cudf::column> probePartToRow;
  std::vector<int64_t> hostRowPartOffset;
  if (probeUseParts) {
    constexpr int32_t kMaxPartsPerRow = 65536;
    rmm::device_scalar<int32_t> partInvalid(0, stream, mr);
    auto parts = extractGeometryPartEnvelopes(
        probeGeomCol,
        emptyExpandCol->view(),
        /*constantExpandBy=*/0.0,
        kMaxPartsPerRow,
        partInvalid.data(),
        stream,
        mr);
    probePartEnv = std::move(parts.envelopes);
    probePartToRow = std::move(parts.partToRow);
    hostRowPartOffset.resize(static_cast<size_t>(numProbeRows) + 1);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostRowPartOffset.data(),
        parts.rowPartOffset->view().data<int64_t>(),
        sizeof(int64_t) * (static_cast<size_t>(numProbeRows) + 1),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
  }

  for (cudf::size_type batchStart = probeRowBegin; batchStart < numProbeRows;) {
    const auto probeBatch =
        std::min<cudf::size_type>(currentBatch, numProbeRows - batchStart);

    std::unique_ptr<cudf::column> probeIndicesAll;
    std::unique_ptr<cudf::column> buildIndicesAll;

    // The cap only makes sense while the batch can still be halved. One
    // continent-scale zone alone sweeps in tens of millions of pickups, and
    // that batch is unsplittable, so give single rows the full int32 headroom
    // and let the downstream candidate chunking bound the work instead.
    const int64_t candidateBudget = probeBatch > 1
        ? kSubBatchCandidateCap
        : std::numeric_limits<int64_t>::max();

    auto sliceCols = [&](GeometryEnvelopes const& src,
                         cudf::size_type lo,
                         cudf::size_type hi) {
      GeometryEnvelopes dst;
      auto sl = [&](std::unique_ptr<cudf::column> const& col) {
        auto slices = cudf::slice(col->view(), {lo, hi}, stream);
        return std::make_unique<cudf::column>(slices[0], stream, mr);
      };
      dst.minX = sl(src.minX);
      dst.minY = sl(src.minY);
      dst.maxX = sl(src.maxX);
      dst.maxY = sl(src.maxY);
      return dst;
    };
    auto remapToRow = [&](std::unique_ptr<cudf::column> partIdx,
                          std::unique_ptr<cudf::column> const& partToRow) {
      AllocLabelGuard allocLabel("spatial.remapToRow");
      auto t = cudf::gather(
          cudf::table_view{{partToRow->view()}},
          partIdx->view(),
          cudf::out_of_bounds_policy::DONT_CHECK,
          stream,
          mr);
      return std::move(t->release()[0]);
    };

    auto generateCandidates = [&]() {
      if (invertPointProbeWithin) {
        // Large polygon build (Q11): grid on this batch of trip points; query
        // with per-ring zone envelopes. Returns (zonePart, localPointIdx).
        auto batchEnvelopes =
            sliceCols(probeEnvelopes, batchStart, batchStart + probeBatch);
        auto pointGrid =
            buildGeometryEnvelopeGrid(std::move(batchEnvelopes), stream, mr);
        GeometryEnvelopes const& zoneQuery =
            spatialIndex_->buildIsPolygonIndex
            ? spatialIndex_->partEnvelopes
            : spatialIndex_->envelopeGrid.envelopes;
        auto [zonePartIdx, pointLocal] = queryGeometryEnvelopeGrid(
            pointGrid, zoneQuery, stream, mr, candidateBudget);
        if (pointLocal->size() == 0) {
          return false;
        }
        cudf::numeric_scalar<cudf::size_type> base(
            batchStart, true, stream, mr);
        auto probeGlobal = cudf::binary_operation(
            pointLocal->view(),
            base,
            cudf::binary_operator::ADD,
            cudf::data_type{cudf::type_to_id<cudf::size_type>()},
            stream,
            mr);
        std::unique_ptr<cudf::column> buildRow =
            spatialIndex_->buildIsPolygonIndex
            ? remapToRow(std::move(zonePartIdx), spatialIndex_->partToRow)
            : std::move(zonePartIdx);
        // A trip point can fall in several rings of one zone; collapse to one
        // (point, zoneRow) pair so COUNT(*) is not inflated.
        auto [pu, bu] = dedupIndexPairs(
            std::move(probeGlobal), std::move(buildRow), stream, mr);
        probeIndicesAll = std::move(pu);
        buildIndicesAll = std::move(bu);
      } else if (probeUseParts) {
        // Q4/Q10: per-ring zone envelopes for this row batch queried against
        // the build point grid. Slice the contiguous part range for the rows.
        const int64_t partLo = hostRowPartOffset[batchStart];
        const int64_t partHi = hostRowPartOffset[batchStart + probeBatch];
        if (partHi <= partLo) {
          return false;
        }
        auto zoneQuery = sliceCols(
            probePartEnv,
            static_cast<cudf::size_type>(partLo),
            static_cast<cudf::size_type>(partHi));
        auto [partLocal, bIdx] = queryGeometryEnvelopeGrid(
            spatialIndex_->envelopeGrid,
            zoneQuery,
            stream,
            mr,
            candidateBudget);
        if (partLocal->size() == 0) {
          return false;
        }
        cudf::numeric_scalar<cudf::size_type> base(
            static_cast<cudf::size_type>(partLo), true, stream, mr);
        auto partGlobal = cudf::binary_operation(
            partLocal->view(),
            base,
            cudf::binary_operator::ADD,
            cudf::data_type{cudf::type_to_id<cudf::size_type>()},
            stream,
            mr);
        auto probeRow = remapToRow(std::move(partGlobal), probePartToRow);
        auto [pu, bu] =
            dedupIndexPairs(std::move(probeRow), std::move(bIdx), stream, mr);
        probeIndicesAll = std::move(pu);
        buildIndicesAll = std::move(bu);
      } else {
        // Default / Q6: probe envelopes against the build-side envelope grid
        // built once in ensureSpatialIndex.
        auto batchEnvelopes =
            sliceCols(probeEnvelopes, batchStart, batchStart + probeBatch);
        auto [pIdx, bIdx] = queryGeometryEnvelopeGrid(
            spatialIndex_->envelopeGrid,
            batchEnvelopes,
            stream,
            mr,
            candidateBudget);
        if (pIdx->size() == 0) {
          return false;
        }
        cudf::numeric_scalar<cudf::size_type> base(
            batchStart, true, stream, mr);
        probeIndicesAll = cudf::binary_operation(
            pIdx->view(),
            base,
            cudf::binary_operator::ADD,
            cudf::data_type{cudf::type_to_id<cudf::size_type>()},
            stream,
            mr);
        buildIndicesAll = std::move(bIdx);
      }
      return true;
    };

    bool hasCandidates = false;
    try {
      AllocLabelGuard allocLabel("spatial.generateCandidates");
      hasCandidates = generateCandidates();
    } catch (std::exception const& e) {
      // Candidate volume is data dependent, so a batch can exceed either the
      // int32 match counter or the free device memory. Both are recoverable by
      // shrinking the batch; only a single probe row is truly unsplittable.
      std::string_view const what(e.what());
      const bool recoverable = what.find("match count overflow") !=
              std::string_view::npos ||
          what.find("out_of_memory") != std::string_view::npos ||
          what.find("out of memory") != std::string_view::npos ||
          what.find("cudaErrorMemoryAllocation") != std::string_view::npos;
      if (probeBatch == 1 || !recoverable) {
        throw;
      }
      batchCeiling = std::max<cudf::size_type>(1, probeBatch / 2);
      currentBatch = std::max<cudf::size_type>(1, probeBatch / 4);
      successStreak = 0;
      continue; // Retry the same probe range with a smaller batch.
    }

    const auto probeOffset = batchStart;
    batchStart += probeBatch;
    if (batchCeiling < kMaxProbeBatch && ++successStreak >= 16) {
      batchCeiling =
          std::min<cudf::size_type>(kMaxProbeBatch, batchCeiling * 2);
      successStreak = 0;
    }
    currentBatch = std::min<cudf::size_type>(
        batchCeiling, std::max<cudf::size_type>(currentBatch * 2, 1));
    if (!hasCandidates) {
      continue;
    }

    auto const totalCandidates = probeIndicesAll->size();

    // A single multi-row sub-batch must not exceed the per-flush cap; if it
    // does, rewind and reprocess this same probe range with fewer rows so no
    // one output slice approaches the column-size limit. Single probe rows are
    // exempt (their matches are bounded by the build row count).
    if (probeBatch > 1 &&
        static_cast<int64_t>(totalCandidates) > kSubBatchCandidateCap) {
      batchStart = probeOffset;
      currentBatch = std::max<cudf::size_type>(1, probeBatch / 2);
      batchCeiling = currentBatch;
      successStreak = 0;
      continue;
    }

    for (cudf::size_type offset = 0; offset < totalCandidates;
         offset += static_cast<cudf::size_type>(candidateChunkRows)) {
      const auto chunkRows = std::min<cudf::size_type>(
          static_cast<cudf::size_type>(candidateChunkRows),
          totalCandidates - offset);
      auto probeSlices = cudf::slice(
          probeIndicesAll->view(),
          {offset, static_cast<cudf::size_type>(offset + chunkRows)},
          stream);
      auto buildSlices = cudf::slice(
          buildIndicesAll->view(),
          {offset, static_cast<cudf::size_type>(offset + chunkRows)},
          stream);
      auto const& probeIndices = probeSlices[0];
      auto const& buildIndices = buildSlices[0];

      rmm::device_scalar<int32_t> distInvalid(0, stream, mr);
      std::unique_ptr<cudf::column> maskOwned;
      ColumnOrView filterColumn;
      cudf::column_view mask;

      const auto pred = spatialPrune_->predicate;
      if (pred == SpatialPrunePredicate::kWithin &&
          spatialPrune_->withinPointOnBuild) {
        // Q4/Q10: zone polygons on probe, points on build. Use pre-extracted
        // point XY from the build envelope grid (minX/minY == point coords)
        // so PIP skips Velox POINT blob parses — dominant win at 600M points.
        // Never gather multipolygon blobs.
        maskOwned = geometryWithinIndexedXY(
            spatialIndex_->envelopeGrid.envelopes.minX->view(),
            spatialIndex_->envelopeGrid.envelopes.minY->view(),
            probeGeomCol,
            buildIndices,
            probeIndices,
            distInvalid.data(),
            stream,
            mr);
        mask = maskOwned->view();
      } else if (
          pred == SpatialPrunePredicate::kWithin &&
          !spatialPrune_->withinPointOnBuild) {
        // Q11 / Q6: points on probe, polygons on build. Point XY comes from
        // probe envelopes (extracted once per getOutput); never gather
        // multipolygon blobs.
        maskOwned = geometryWithinIndexedXY(
            probeEnvelopes.minX->view(),
            probeEnvelopes.minY->view(),
            buildGeomCol,
            probeIndices,
            buildIndices,
            distInvalid.data(),
            stream,
            mr);
        mask = maskOwned->view();
      } else if (pred == SpatialPrunePredicate::kDistanceLE &&
          buildRadiusChannel_.has_value()) {
        AllocLabelGuard allocLabel("spatial.distanceRadius");
        auto gatheredProbeGeom = cudf::gather(
            cudf::table_view{{probeGeomCol}},
            probeIndices,
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);
        auto gatheredBuildGeom = cudf::gather(
            cudf::table_view{{buildGeomCol}},
            buildIndices,
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);
        auto distances = geometryDistance(
            gatheredProbeGeom->view().column(0),
            gatheredBuildGeom->view().column(0),
            distInvalid.data(),
            stream,
            mr);
        // Invalid pairs are nulled in the distance column.
        auto gatheredRadius = cudf::gather(
            cudf::table_view{{buildView.column(buildRadiusChannel_.value())}},
            buildIndices,
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);
        maskOwned = cudf::binary_operation(
            distances->view(),
            gatheredRadius->view().column(0),
            cudf::binary_operator::LESS_EQUAL,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            mr);
        mask = maskOwned->view();
      } else if (pred == SpatialPrunePredicate::kWithin) {
        AllocLabelGuard allocLabel("spatial.withinFallback");
        auto gatheredProbeGeom = cudf::gather(
            cudf::table_view{{probeGeomCol}},
            probeIndices,
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);
        auto gatheredBuildGeom = cudf::gather(
            cudf::table_view{{buildGeomCol}},
            buildIndices,
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);
        cudf::column_view pointView;
        cudf::column_view polyView;
        if (spatialPrune_->withinPointOnBuild) {
          pointView = gatheredBuildGeom->view().column(0);
          polyView = gatheredProbeGeom->view().column(0);
        } else {
          pointView = gatheredProbeGeom->view().column(0);
          polyView = gatheredBuildGeom->view().column(0);
        }
        maskOwned = geometryWithin(
            pointView, polyView, distInvalid.data(), stream, mr);
        mask = maskOwned->view();
      } else if (pred == SpatialPrunePredicate::kIntersects) {
        AllocLabelGuard allocLabel("spatial.intersects");
        auto gatheredProbeGeom = cudf::gather(
            cudf::table_view{{probeGeomCol}},
            probeIndices,
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);
        auto gatheredBuildGeom = cudf::gather(
            cudf::table_view{{buildGeomCol}},
            buildIndices,
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);
        maskOwned = geometryIntersects(
            gatheredProbeGeom->view().column(0),
            gatheredBuildGeom->view().column(0),
            distInvalid.data(),
            stream,
            mr);
        // Invalid pairs are nulled; do not fail the join batch.
        mask = maskOwned->view();
      } else {
        // Distance without radius, or unknown: fall back to full filter eval.
        AllocLabelGuard allocLabel("spatial.fullFilterFallback");
        auto gatheredProbe = cudf::gather(
            probeTableView,
            probeIndices,
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);
        auto gatheredBuild = cudf::gather(
            buildView,
            buildIndices,
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);
        std::vector<cudf::column_view> combinedViews;
        auto gatheredProbeView = gatheredProbe->view();
        auto gatheredBuildView = gatheredBuild->view();
        combinedViews.reserve(
            gatheredProbeView.num_columns() + gatheredBuildView.num_columns());
        for (cudf::size_type i = 0; i < gatheredProbeView.num_columns(); ++i) {
          combinedViews.push_back(gatheredProbeView.column(i));
        }
        for (cudf::size_type i = 0; i < gatheredBuildView.num_columns(); ++i) {
          combinedViews.push_back(gatheredBuildView.column(i));
        }
        filterColumn = filterEvaluator_->eval(combinedViews, stream, mr);
        mask = asView(filterColumn);
      }

      // Compound AND: specialized Within/Intersects kernels only cover one
      // conjunct. Envelope AABB prefilter is necessary but not sufficient for
      // ST_Intersects(const_poly, zone), so always refine with the full filter.
      if (spatialPrune_->filterIsCompound && maskOwned) {
        auto passedProbe = cudf::apply_boolean_mask(
            cudf::table_view{{probeIndices}}, mask, stream, mr);
        auto passedBuild = cudf::apply_boolean_mask(
            cudf::table_view{{buildIndices}}, mask, stream, mr);
        auto pCols = passedProbe->release();
        auto bCols = passedBuild->release();
        const auto passedRows = pCols[0]->size();
        if (passedRows == 0) {
          continue;
        }

        // Refining materializes every filter input once per surviving pair, so
        // a geometry blob column duplicates multi-KB rows: at SF100 Q11 that is
        // a single ~14GB gather. Slice the survivors to bound it by bytes.
        AllocLabelGuard allocLabel("spatial.compoundRefine");

        // The indexed kernel already decided Within exactly, so when the other
        // conjuncts are known not to read a column we can hand the evaluator an
        // empty placeholder for it. That keeps multi-KB geometry blobs out of
        // the gather entirely instead of merely chunking them.
        const bool useResidual = residualEvaluator_ != nullptr &&
            spatialPrune_->predicate == SpatialPrunePredicate::kWithin;
        std::vector<cudf::size_type> probeKeep;
        std::vector<cudf::size_type> buildKeep;
        if (useResidual) {
          const auto probeCols = probeTableView.num_columns();
          for (cudf::size_type i = 0; i < probeCols; ++i) {
            const bool wide =
                probeTableView.column(i).type().id() == cudf::type_id::STRING;
            if (!wide || !residualUnreadColumn_[i]) {
              probeKeep.push_back(i);
            }
          }
          for (cudf::size_type i = 0; i < buildView.num_columns(); ++i) {
            const bool wide =
                buildView.column(i).type().id() == cudf::type_id::STRING;
            if (!wide || !residualUnreadColumn_[probeCols + i]) {
              buildKeep.push_back(i);
            }
          }
          LOG_FIRST_N(WARNING, 5)
              << "[geo:refine] residual-only gather probeKeep="
              << probeKeep.size() << "/" << probeCols
              << " buildKeep=" << buildKeep.size() << "/"
              << buildView.num_columns();
        }
        // The mean row size cannot bound this gather: geometry sizes are heavy
        // tailed (a 5MB continent-scale zone against a 5KB mean), and the
        // skewed tail is exactly where the same huge polygons repeat across
        // candidates. Start from the mean and back off on OOM instead.
        constexpr int64_t kRefineByteBudget = int64_t{1} << 30;
        auto refineChunk = useResidual
            ? passedRows
            : static_cast<cudf::size_type>(std::clamp<int64_t>(
                  kRefineByteBudget / refineRowBytes, 1024, passedRows));
        LOG_FIRST_N(WARNING, 5)
            << "[geo:gather] site=compoundRefine passed=" << passedRows
            << " refineChunk=" << refineChunk
            << " refineRowBytes=" << refineRowBytes;

        for (cudf::size_type refineOffset = 0; refineOffset < passedRows;) {
          auto refineRows =
              std::min<cudf::size_type>(refineChunk, passedRows - refineOffset);
          while (true) {
            try {
              auto probeSlice = cudf::slice(
                  pCols[0]->view(),
                  {refineOffset, refineOffset + refineRows},
                  stream)[0];
              auto buildSlice = cudf::slice(
                  bCols[0]->view(),
                  {refineOffset, refineOffset + refineRows},
                  stream)[0];

              auto gatheredProbe = cudf::gather(
                  useResidual ? probeTableView.select(probeKeep)
                              : probeTableView,
                  probeSlice,
                  cudf::out_of_bounds_policy::DONT_CHECK,
                  stream,
                  mr);
              auto gatheredBuild = cudf::gather(
                  useResidual ? buildView.select(buildKeep) : buildView,
                  buildSlice,
                  cudf::out_of_bounds_policy::DONT_CHECK,
                  stream,
                  mr);
              std::vector<cudf::column_view> combinedViews;
              auto gp = gatheredProbe->view();
              auto gb = gatheredBuild->view();
              std::unique_ptr<cudf::column> placeholder;
              if (useResidual) {
                // Positions must match the evaluator's schema, so reinstate the
                // skipped columns as empty strings of the same length.
                placeholder = cudf::make_column_from_scalar(
                    cudf::string_scalar("", true, stream, mr),
                    refineRows,
                    stream,
                    mr);
                combinedViews.reserve(
                    probeTableView.num_columns() + buildView.num_columns());
                cudf::size_type next = 0;
                for (cudf::size_type i = 0; i < probeTableView.num_columns();
                     ++i) {
                  if (next < static_cast<cudf::size_type>(probeKeep.size()) &&
                      probeKeep[next] == i) {
                    combinedViews.push_back(gp.column(next));
                    ++next;
                  } else {
                    combinedViews.push_back(placeholder->view());
                  }
                }
                next = 0;
                for (cudf::size_type i = 0; i < buildView.num_columns(); ++i) {
                  if (next < static_cast<cudf::size_type>(buildKeep.size()) &&
                      buildKeep[next] == i) {
                    combinedViews.push_back(gb.column(next));
                    ++next;
                  } else {
                    combinedViews.push_back(placeholder->view());
                  }
                }
              } else {
                combinedViews.reserve(gp.num_columns() + gb.num_columns());
                for (cudf::size_type i = 0; i < gp.num_columns(); ++i) {
                  combinedViews.push_back(gp.column(i));
                }
                for (cudf::size_type i = 0; i < gb.num_columns(); ++i) {
                  combinedViews.push_back(gb.column(i));
                }
              }
              filterColumn = useResidual
                  ? residualEvaluator_->eval(combinedViews, stream, mr)
                  : filterEvaluator_->eval(combinedViews, stream, mr);
              auto refined = cudf::apply_boolean_mask(
                  cudf::table_view{{probeSlice}},
                  asView(filterColumn),
                  stream,
                  mr);
              auto refinedProbe = refined->release();
              if (refinedProbe[0]->size() > 0) {
                accumulatedMatches += refinedProbe[0]->size();
                matchedProbeChunks.push_back(std::move(refinedProbe[0]));
                if (needBuildIndices) {
                  auto refinedBuild = cudf::apply_boolean_mask(
                      cudf::table_view{{buildSlice}},
                      asView(filterColumn),
                      stream,
                      mr);
                  matchedBuildChunks.push_back(
                      std::move(refinedBuild->release()[0]));
                }
              }
              break;
            } catch (std::exception const& e) {
              std::string_view const what(e.what());
              const bool oom =
                  what.find("out_of_memory") != std::string_view::npos ||
                  what.find("out of memory") != std::string_view::npos ||
                  what.find("Maximum pool size exceeded") !=
                      std::string_view::npos ||
                  what.find("cudaErrorMemoryAllocation") !=
                      std::string_view::npos;
              if (!oom || refineRows <= 1) {
                throw;
              }
              refineRows = std::max<cudf::size_type>(1, refineRows / 4);
              // Keep the smaller size for the remaining chunks: skew persists.
              refineChunk = refineRows;
              LOG_FIRST_N(WARNING, 20)
                  << "[geo:gather] compoundRefine backoff rows=" << refineRows;
            }
          }
          refineOffset += refineRows;
        }
        continue;
      }

      auto filteredProbeIndices = cudf::apply_boolean_mask(
          cudf::table_view{{probeIndices}}, mask, stream, mr);
      auto probeIndicesCols = filteredProbeIndices->release();
      if (probeIndicesCols[0]->size() == 0) {
        continue;
      }
      accumulatedMatches += probeIndicesCols[0]->size();
      matchedProbeChunks.push_back(std::move(probeIndicesCols[0]));

      if (needBuildIndices) {
        auto filteredBuildIndices = cudf::apply_boolean_mask(
            cudf::table_view{{buildIndices}}, mask, stream, mr);
        matchedBuildChunks.push_back(
            std::move(filteredBuildIndices->release()[0]));
      }
    } // candidate chunks

    // Flush at a sub-batch boundary once we have enough matched rows for one
    // output slice; resume from batchStart on the next call.
    if (probeRowsConsumed && batchStart < numProbeRows &&
        accumulatedMatches >= outputRowBudget) {
      consumedRows = batchStart;
      break;
    }
  } // probe envelope batches

  if (probeRowsConsumed) {
    *probeRowsConsumed = consumedRows;
  }

  if (matchedProbeChunks.empty()) {
    return {
        cudf::make_empty_column(cudf::type_to_id<cudf::size_type>()),
        needBuildIndices
            ? cudf::make_empty_column(cudf::type_to_id<cudf::size_type>())
            : nullptr};
  }

  std::vector<cudf::column_view> probeViews;
  probeViews.reserve(matchedProbeChunks.size());
  for (auto& chunk : matchedProbeChunks) {
    probeViews.push_back(chunk->view());
  }
  auto concatProbe = cudf::concatenate(probeViews, stream, mr);

  std::unique_ptr<cudf::column> concatBuild;
  if (needBuildIndices) {
    std::vector<cudf::column_view> buildViews;
    buildViews.reserve(matchedBuildChunks.size());
    for (auto& chunk : matchedBuildChunks) {
      buildViews.push_back(chunk->view());
    }
    concatBuild = cudf::concatenate(buildViews, stream, mr);
  }
  return {std::move(concatProbe), std::move(concatBuild)};
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
CudfNestedLoopJoinProbe::crossJoinConditionalIndices(
    cudf::table_view probeTableView,
    cudf::table_view buildView,
    rmm::cuda_stream_view stream,
    bool needBuildIndices,
    cudf::size_type probeRowBegin,
    cudf::size_type* probeRowsConsumed) {
  if (spatialPrune_.has_value()) {
    return spatialPruneConditionalIndices(
        probeTableView,
        buildView,
        stream,
        needBuildIndices,
        probeRowBegin,
        probeRowsConsumed);
  }
  // Non-spatial cross-product path processes the whole probe input at once.
  if (probeRowsConsumed) {
    *probeRowsConsumed = probeTableView.num_rows();
  }
  VELOX_NVTX_FUNC_RANGE();
  auto mr = get_temp_mr();

  const auto numProbeRows = probeTableView.num_rows();
  const auto numBuildRows = buildView.num_rows();
  if (numProbeRows == 0 || numBuildRows == 0) {
    return {
        cudf::make_empty_column(cudf::type_to_id<cudf::size_type>()),
        needBuildIndices
            ? cudf::make_empty_column(cudf::type_to_id<cudf::size_type>())
            : nullptr};
  }

  VELOX_CHECK_NOT_NULL(
      filterEvaluator_,
      "Join filter evaluator must be initialized before "
      "crossJoinConditionalIndices");

  // Tile the build side so we never materialize a full probe×build cross
  // product (SpatialBench Q8 is ~6M×20K at SF1). Cap tile product size; keep
  // at least one build row per tile.
  constexpr int64_t kMaxCrossProductRows = 4'000'000;
  const auto buildChunkRows = std::max<cudf::size_type>(
      1,
      static_cast<cudf::size_type>(std::min<int64_t>(
          numBuildRows, kMaxCrossProductRows / numProbeRows)));

  auto zero = cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr);
  auto one = cudf::numeric_scalar<cudf::size_type>(1, true, stream, mr);
  auto probeRange = cudf::sequence(numProbeRows, zero, one, stream, mr);

  std::vector<std::unique_ptr<cudf::column>> matchedProbeChunks;
  std::vector<std::unique_ptr<cudf::column>> matchedBuildChunks;
  matchedProbeChunks.reserve(
      (static_cast<size_t>(numBuildRows) + buildChunkRows - 1) /
      buildChunkRows);
  if (needBuildIndices) {
    matchedBuildChunks.reserve(matchedProbeChunks.capacity());
  }

  for (cudf::size_type buildOffset = 0; buildOffset < numBuildRows;
       buildOffset += buildChunkRows) {
    const auto chunkRows =
        std::min(buildChunkRows, numBuildRows - buildOffset);
    const auto totalRows =
        static_cast<int64_t>(numProbeRows) * static_cast<int64_t>(chunkRows);
    VELOX_CHECK_LE(
        totalRows,
        std::numeric_limits<cudf::size_type>::max(),
        "Cross product tile exceeds cudf::size_type limit: {} x {} = {} rows",
        numProbeRows,
        chunkRows,
        totalRows);

    auto buildChunkSlices = cudf::slice(
        buildView,
        {buildOffset, static_cast<cudf::size_type>(buildOffset + chunkRows)},
        stream);
    auto const& buildChunk = buildChunkSlices[0];

    auto probeIndicesTable = cudf::repeat(
        cudf::table_view{{probeRange->view()}}, chunkRows, stream, mr);
    auto buildLocalRange = cudf::sequence(chunkRows, zero, one, stream, mr);
    auto buildLocalIndicesTable = cudf::tile(
        cudf::table_view{{buildLocalRange->view()}}, numProbeRows, stream, mr);
    auto probeIndices = std::move(probeIndicesTable->release()[0]);
    auto buildLocalIndices = std::move(buildLocalIndicesTable->release()[0]);

    AllocLabelGuard allocLabel("crossJoin.tile");
    auto gatheredProbe = cudf::gather(
        probeTableView,
        probeIndices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        mr);
    auto gatheredBuild = cudf::gather(
        buildChunk,
        buildLocalIndices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        mr);

    std::vector<cudf::column_view> combinedViews;
    auto gatheredProbeView = gatheredProbe->view();
    auto gatheredBuildView = gatheredBuild->view();
    combinedViews.reserve(
        gatheredProbeView.num_columns() + gatheredBuildView.num_columns());
    for (cudf::size_type i = 0; i < gatheredProbeView.num_columns(); ++i) {
      combinedViews.push_back(gatheredProbeView.column(i));
    }
    for (cudf::size_type i = 0; i < gatheredBuildView.num_columns(); ++i) {
      combinedViews.push_back(gatheredBuildView.column(i));
    }

    auto filterColumn = filterEvaluator_->eval(combinedViews, stream, mr);
    auto mask = asView(filterColumn);

    auto filteredProbeIndices = cudf::apply_boolean_mask(
        cudf::table_view{{probeIndices->view()}}, mask, stream, mr);
    auto probeIndicesCols = filteredProbeIndices->release();
    if (probeIndicesCols[0]->size() == 0) {
      continue;
    }
    matchedProbeChunks.push_back(std::move(probeIndicesCols[0]));

    if (needBuildIndices) {
      // Convert chunk-local build indices to global by adding buildOffset.
      auto offsetScalar = cudf::numeric_scalar<cudf::size_type>(
          buildOffset, true, stream, mr);
      auto globalBuildIndices = cudf::binary_operation(
          buildLocalIndices->view(),
          offsetScalar,
          cudf::binary_operator::ADD,
          cudf::data_type{cudf::type_to_id<cudf::size_type>()},
          stream,
          mr);
      auto filteredBuildIndices = cudf::apply_boolean_mask(
          cudf::table_view{{globalBuildIndices->view()}}, mask, stream, mr);
      matchedBuildChunks.push_back(
          std::move(filteredBuildIndices->release()[0]));
    }
  }

  if (matchedProbeChunks.empty()) {
    return {
        cudf::make_empty_column(cudf::type_to_id<cudf::size_type>()),
        needBuildIndices
            ? cudf::make_empty_column(cudf::type_to_id<cudf::size_type>())
            : nullptr};
  }

  std::vector<cudf::column_view> probeViews;
  probeViews.reserve(matchedProbeChunks.size());
  for (auto& chunk : matchedProbeChunks) {
    probeViews.push_back(chunk->view());
  }
  auto concatProbe = cudf::concatenate(probeViews, stream, mr);

  std::unique_ptr<cudf::column> concatBuild;
  if (needBuildIndices) {
    std::vector<cudf::column_view> buildViews;
    buildViews.reserve(matchedBuildChunks.size());
    for (auto& chunk : matchedBuildChunks) {
      buildViews.push_back(chunk->view());
    }
    concatBuild = cudf::concatenate(buildViews, stream, mr);
  }
  return {std::move(concatProbe), std::move(concatBuild)};
}

CudfNestedLoopJoinProbe::JoinOutput
CudfNestedLoopJoinProbe::joinWithBuildBatch(
    cudf::table_view probeTableView,
    cudf::table_view buildView,
    rmm::cuda_stream_view stream,
    cudf::size_type probeRowBegin,
    cudf::size_type* probeRowsConsumed) {
  VELOX_NVTX_FUNC_RANGE();

  syncBuildStream(stream);

  auto numOutputColumns = outputType_->size();

  // Extend probe view with precomputed columns for filter AST evaluation.
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedProbeView = probeTableView;
  if (hasFilter_ && useAstFilter_ && !leftPrecomputeInstructions_.empty()) {
    auto probeColumnViews = tableViewToColumnViews(probeTableView);
    leftPrecomputed = precomputeSubexpressions(
        probeColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedProbeView =
        createExtendedTableView(probeTableView, leftPrecomputed);
  }
  // Use cached extended build view if build-side precompute was needed.
  const cudf::table_view& extendedBuildView =
      buildPrecomputed_.empty() ? buildView : buildExtendedView_;

  if (hasFilter_) {
    VELOX_CHECK(
        isInitialized(),
        "Filter must be initialized before joinWithBuildBatch");

    // Owning storage for whichever path below produces the index pairs;
    // leftIndicesView/rightIndicesView alias into one of these two.
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftIndicesBuffer;
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> rightIndicesBuffer;
    std::unique_ptr<cudf::column> leftIndicesColumn;
    std::unique_ptr<cudf::column> rightIndicesColumn;
    cudf::column_view leftIndicesView;
    cudf::column_view rightIndicesView;

    if (useAstFilter_) {
      // AST joins are not streamed; the whole probe input is consumed here.
      if (probeRowsConsumed) {
        *probeRowsConsumed = probeTableView.num_rows();
      }
      std::tie(leftIndicesBuffer, rightIndicesBuffer) =
          cudf::conditional_inner_join(
              extendedProbeView,
              extendedBuildView,
              tree_.back(),
              std::nullopt,
              stream,
              get_temp_mr());

      VELOX_CHECK_LE(
          static_cast<int64_t>(leftIndicesBuffer->size()),
          std::numeric_limits<cudf::size_type>::max(),
          "Conditional join output exceeds cudf::size_type limit: {} rows",
          leftIndicesBuffer->size());

      leftIndicesView = cudf::column_view(
          cudf::data_type{cudf::type_to_id<cudf::size_type>()},
          leftIndicesBuffer->size(),
          leftIndicesBuffer->data(),
          nullptr,
          0);
      rightIndicesView = cudf::column_view(
          cudf::data_type{cudf::type_to_id<cudf::size_type>()},
          rightIndicesBuffer->size(),
          rightIndicesBuffer->data(),
          nullptr,
          0);
    } else {
      // Condition spans both sides with a non-AST sub-expression; evaluate
      // it generally against the full cross product instead of driving
      // cudf::conditional_inner_join with an AST tree.
      std::tie(leftIndicesColumn, rightIndicesColumn) =
          crossJoinConditionalIndices(
              probeTableView,
              buildView,
              stream,
              /*needBuildIndices=*/true,
              probeRowBegin,
              probeRowsConsumed);
      leftIndicesView = leftIndicesColumn->view();
      rightIndicesView = rightIndicesColumn->view();
    }

    // Track which probe rows matched for left/full join mismatch handling.
    // Uses cudf::contains to check which probe row indices [0..N) appear
    // in the join result.
    if (isLeftOrFullJoin()) {
      auto numProbeRows = probeTableView.num_rows();
      auto probeRowSequence = cudf::sequence(
          numProbeRows,
          cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
          stream,
          get_temp_mr());

      // The spatial path may stream one probe input across several slices
      // (see doGetOutput). Each slice reports which probe rows it matched;
      // OR them so mismatch emission sees matches from every slice.
      auto matchedThisSlice = cudf::contains(
          leftIndicesView, probeRowSequence->view(), stream, get_temp_mr());
      if (probeMatchedFlags_) {
        probeMatchedFlags_ = cudf::binary_operation(
            probeMatchedFlags_->view(),
            matchedThisSlice->view(),
            cudf::binary_operator::BITWISE_OR,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            get_temp_mr());
      } else {
        probeMatchedFlags_ = std::move(matchedThisSlice);
      }
    }

    // Track which build rows matched for right/full join mismatch handling.
    if (isRightOrFullJoin()) {
      auto numBuildRows = buildView.num_rows();
      auto buildRowSequence = cudf::sequence(
          numBuildRows,
          cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
          stream,
          get_temp_mr());

      auto matchedInBatch = cudf::contains(
          rightIndicesView, buildRowSequence->view(), stream, get_temp_mr());

      auto updatedFlags = cudf::binary_operation(
          buildMatchedFlags_->view(),
          matchedInBatch->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          get_temp_mr());
      stream.synchronize();
      buildMatchedFlags_ = std::move(updatedFlags);
    }

    // Gather only the columns needed for output.
    // Zero-column output (COUNT(*)): skip gathers; row count comes from indices.
    if (numOutputColumns == 0) {
      return {
          std::make_unique<cudf::table>(),
          static_cast<vector_size_t>(leftIndicesView.size())};
    }

    auto probeGatherView = probeTableView.select(probeColumnIndicesToGather_);
    auto buildGatherView = buildView.select(buildColumnIndicesToGather_);

    AllocLabelGuard allocLabel("join.buildOutput");
    auto gatheredProbe = cudf::gather(
        probeGatherView,
        leftIndicesView,
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        get_output_mr());

    auto gatheredBuild = cudf::gather(
        buildGatherView,
        rightIndicesView,
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        get_output_mr());

    std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);
    auto probeCols = gatheredProbe->release();
    auto buildCols = gatheredBuild->release();
    for (size_t i = 0; i < probeColumnOutputIndices_.size(); ++i) {
      outCols[probeColumnOutputIndices_[i]] = std::move(probeCols[i]);
    }
    for (size_t i = 0; i < buildColumnOutputIndices_.size(); ++i) {
      outCols[buildColumnOutputIndices_[i]] = std::move(buildCols[i]);
    }

    // leftIndicesView.size() is the match count; table->num_rows() is 0 when
    // the output schema is empty (COUNT(*)).
    return {
        std::make_unique<cudf::table>(std::move(outCols)),
        static_cast<vector_size_t>(leftIndicesView.size())};
  }

  // Unfiltered join using cross_join.
  auto outputRows = static_cast<int64_t>(probeTableView.num_rows()) *
      static_cast<int64_t>(buildView.num_rows());
  VELOX_CHECK_LE(
      outputRows,
      std::numeric_limits<cudf::size_type>::max(),
      "Cross join output exceeds cudf::size_type limit: {} x {} = {} rows",
      probeTableView.num_rows(),
      buildView.num_rows(),
      outputRows);

  if (numOutputColumns == 0) {
    return {
        std::make_unique<cudf::table>(),
        static_cast<vector_size_t>(outputRows)};
  }

  auto crossResult =
      cudf::cross_join(probeTableView, buildView, stream, get_output_mr());

  // Cross join matches every row, so no per-row matched flags are needed:
  // probeMatchedFlags_ is only consumed via emitProbeMismatchRows, which is
  // unreachable in the unfiltered path (see doGetOutput). buildMatchedFlags_
  // is skipped in isBlocked for !hasFilter_; emitBuildMismatchRows early-
  // returns in that case.

  auto allCols = crossResult->release();
  auto numProbeCols = probeTableView.num_columns();

  std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);
  for (size_t i = 0; i < probeColumnOutputIndices_.size(); ++i) {
    outCols[probeColumnOutputIndices_[i]] =
        std::move(allCols[probeColumnIndicesToGather_[i]]);
  }
  for (size_t i = 0; i < buildColumnOutputIndices_.size(); ++i) {
    outCols[buildColumnOutputIndices_[i]] =
        std::move(allCols[numProbeCols + buildColumnIndicesToGather_[i]]);
  }

  return {
      std::make_unique<cudf::table>(std::move(outCols)),
      static_cast<vector_size_t>(outputRows)};
}

CudfNestedLoopJoinProbe::JoinOutput
CudfNestedLoopJoinProbe::emitProbeMismatchRows(
    cudf::table_view probeTableView,
    rmm::cuda_stream_view stream) {
  auto probeGatherView = probeTableView.select(probeColumnIndicesToGather_);

  std::unique_ptr<cudf::table> unmatchedProbe;
  vector_size_t numUnmatched = 0;
  if (!probeMatchedFlags_) {
    // No flags means all probe rows are unmatched (empty build case).
    // When probe gathers nothing (empty output schema), use probe row count.
    if (probeColumnIndicesToGather_.empty()) {
      numUnmatched = static_cast<vector_size_t>(probeTableView.num_rows());
      unmatchedProbe = std::make_unique<cudf::table>();
    } else {
      unmatchedProbe = std::make_unique<cudf::table>(
          probeGatherView, stream, get_output_mr());
      numUnmatched = static_cast<vector_size_t>(unmatchedProbe->num_rows());
    }
  } else {
    auto unmatchedMask = cudf::unary_operation(
        probeMatchedFlags_->view(),
        cudf::unary_operator::NOT,
        stream,
        get_temp_mr());
    if (probeColumnIndicesToGather_.empty()) {
      // Zero-column output: count unmatched via a throwaway index column.
      auto seq = cudf::sequence(
          probeMatchedFlags_->size(),
          cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
          stream,
          get_temp_mr());
      auto filtered = cudf::apply_boolean_mask(
          cudf::table_view{{seq->view()}},
          unmatchedMask->view(),
          stream,
          get_temp_mr());
      numUnmatched = static_cast<vector_size_t>(filtered->num_rows());
      unmatchedProbe = std::make_unique<cudf::table>();
    } else {
      unmatchedProbe = cudf::apply_boolean_mask(
          probeGatherView, unmatchedMask->view(), stream, get_output_mr());
      numUnmatched = static_cast<vector_size_t>(unmatchedProbe->num_rows());
    }
  }

  if (numUnmatched == 0) {
    return {nullptr, 0};
  }

  auto numOutputColumns = outputType_->size();
  std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);

  // Place unmatched probe columns at their output positions.
  if (!probeColumnIndicesToGather_.empty()) {
    auto probeCols = unmatchedProbe->release();
    for (size_t i = 0; i < probeColumnOutputIndices_.size(); ++i) {
      outCols[probeColumnOutputIndices_[i]] = std::move(probeCols[i]);
    }
  }

  // Create all-null columns for the build side.
  for (size_t i = 0; i < buildColumnOutputIndices_.size(); ++i) {
    auto outIdx = buildColumnOutputIndices_[i];
    auto buildChannel = buildColumnIndicesToGather_[i];
    auto buildCudfDataType =
        veloxToCudfDataType(buildType_->childAt(buildChannel));
    auto nullScalar = cudf::make_default_constructed_scalar(
        buildCudfDataType, stream, get_temp_mr());
    outCols[outIdx] = cudf::make_column_from_scalar(
        *nullScalar, numUnmatched, stream, get_output_mr());
  }

  return {
      std::make_unique<cudf::table>(std::move(outCols)), numUnmatched};
}

RowVectorPtr CudfNestedLoopJoinProbe::emitBuildMismatchRows(
    rmm::cuda_stream_view stream) {
  // Unfiltered cross_join already emitted every build row, so no mismatches
  // to emit. buildMatchedFlags_ is not allocated in that case.
  if (!buildMatchedFlags_) {
    finished_ = true;
    return nullptr;
  }
  auto& buildTable = buildData_.value();
  auto numOutputColumns = outputType_->size();

  // Invert flags: unmatched = NOT(matched).
  auto unmatchedMask = cudf::unary_operation(
      buildMatchedFlags_->view(),
      cudf::unary_operator::NOT,
      stream,
      get_temp_mr());

  // Select unmatched build rows.
  auto buildGatherView = buildTable->view().select(buildColumnIndicesToGather_);
  vector_size_t numUnmatched = 0;
  std::unique_ptr<cudf::table> unmatchedBuild;
  if (buildColumnIndicesToGather_.empty()) {
    auto seq = cudf::sequence(
        unmatchedMask->size(),
        cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
        cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
        stream,
        get_temp_mr());
    auto filtered = cudf::apply_boolean_mask(
        cudf::table_view{{seq->view()}},
        unmatchedMask->view(),
        stream,
        get_temp_mr());
    numUnmatched = static_cast<vector_size_t>(filtered->num_rows());
    unmatchedBuild = std::make_unique<cudf::table>();
  } else {
    unmatchedBuild = cudf::apply_boolean_mask(
        buildGatherView, unmatchedMask->view(), stream, get_output_mr());
    numUnmatched = static_cast<vector_size_t>(unmatchedBuild->num_rows());
  }

  finished_ = true;
  if (numUnmatched == 0) {
    return nullptr;
  }

  std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);

  // Create all-null columns for the probe side.
  for (size_t li = 0; li < probeColumnOutputIndices_.size(); ++li) {
    auto outIdx = probeColumnOutputIndices_[li];
    auto probeChannel = probeColumnIndicesToGather_[li];
    auto probeCudfDataType =
        veloxToCudfDataType(probeType_->childAt(probeChannel));
    auto nullScalar = cudf::make_default_constructed_scalar(
        probeCudfDataType, stream, get_temp_mr());
    outCols[outIdx] = cudf::make_column_from_scalar(
        *nullScalar, numUnmatched, stream, get_output_mr());
  }

  // Place unmatched build columns at their output positions.
  if (!buildColumnIndicesToGather_.empty()) {
    auto buildCols = unmatchedBuild->release();
    for (size_t ri = 0; ri < buildColumnOutputIndices_.size(); ++ri) {
      outCols[buildColumnOutputIndices_[ri]] = std::move(buildCols[ri]);
    }
  }

  auto out = std::make_unique<cudf::table>(std::move(outCols));
  // Zero-column output cannot encode row count in the cudf table.
  auto size = outputType_->size() == 0
      ? numUnmatched
      : static_cast<vector_size_t>(out->num_rows());
  return std::make_shared<CudfVector>(
      operatorCtx_->pool(), outputType_, size, std::move(out), stream);
}

RowVectorPtr CudfNestedLoopJoinProbe::doGetOutput() {
  if (!input_) {
    // Right/full join: after all probe inputs, the last driver emits
    // unmatched build rows with null probe columns.
    if (isRightOrFullJoin() && noMoreInput_ && isLastDriver_ &&
        !buildMismatchEmitted_) {
      buildMismatchEmitted_ = true;
      auto stream = cudfGlobalStreamPool().get_stream();
      return emitBuildMismatchRows(stream);
    }
    if (noMoreInput_) {
      finished_ = true;
    }
    return nullptr;
  }

  VELOX_CHECK(buildData_.has_value(), "Build data not available in getOutput");
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput);
  auto stream = cudfInput->stream();
  lastProbeStream_ = stream;

  // LeftSemiProject: emit all probe rows with a boolean match column.
  if (joinType_ == core::JoinType::kLeftSemiProject) {
    auto probeTableView = cudfInput->getTableView();
    auto numProbeRows = static_cast<cudf::size_type>(probeTableView.num_rows());

    std::unique_ptr<cudf::column> matchFlags;
    if (buildEmpty_ || !hasFilter_) {
      // No filter + non-empty build: all probe rows match (true).
      // Empty build: no probe rows match (false).
      auto scalar =
          cudf::numeric_scalar<bool>(!buildEmpty_, true, stream, get_temp_mr());
      matchFlags = cudf::make_column_from_scalar(
          scalar, numProbeRows, stream, get_temp_mr());
    } else {
      // Filtered: compute matched probe indices against the single build table.
      auto falseScalar =
          cudf::numeric_scalar<bool>(false, true, stream, get_temp_mr());
      matchFlags = cudf::make_column_from_scalar(
          falseScalar, numProbeRows, stream, get_temp_mr());

      // Owning storage for whichever path below produces the matched probe
      // indices; matchedIndicesView aliases into one of these two.
      std::unique_ptr<rmm::device_uvector<cudf::size_type>>
          matchedIndicesBuffer;
      std::unique_ptr<cudf::column> matchedIndicesColumn;
      cudf::size_type matchedIndicesSize = 0;
      cudf::column_view matchedIndicesView;

      if (useAstFilter_) {
        // Extend probe view with precomputed columns if needed.
        std::vector<ColumnOrView> leftPrecomputed;
        cudf::table_view extendedProbeView = probeTableView;
        if (!leftPrecomputeInstructions_.empty()) {
          auto probeColumnViews = tableViewToColumnViews(probeTableView);
          leftPrecomputed = precomputeSubexpressions(
              probeColumnViews,
              leftPrecomputeInstructions_,
              scalars_,
              probeType_,
              stream);
          extendedProbeView =
              createExtendedTableView(probeTableView, leftPrecomputed);
        }
        const cudf::table_view& extendedBuildView = buildPrecomputed_.empty()
            ? buildData_.value()->view()
            : buildExtendedView_;

        matchedIndicesBuffer = cudf::conditional_left_semi_join(
            extendedProbeView,
            extendedBuildView,
            tree_.back(),
            {},
            stream,
            get_temp_mr());
        matchedIndicesSize =
            static_cast<cudf::size_type>(matchedIndicesBuffer->size());
        matchedIndicesView = cudf::column_view(
            cudf::data_type{cudf::type_to_id<cudf::size_type>()},
            matchedIndicesBuffer->size(),
            matchedIndicesBuffer->data(),
            nullptr,
            0);
      } else {
        // Condition spans both sides with a non-AST sub-expression; a probe
        // row "matches" (for the semi-join match flag) if it appears at all
        // among the filtered cross-product probe indices. Build indices
        // aren't needed here, so skip computing them.
        auto [probeIndicesForSemiJoin, unusedBuildIndices] =
            crossJoinConditionalIndices(
                probeTableView,
                buildData_.value()->view(),
                stream,
                /*needBuildIndices=*/false);
        matchedIndicesColumn = std::move(probeIndicesForSemiJoin);
        matchedIndicesSize = matchedIndicesColumn->size();
        matchedIndicesView = matchedIndicesColumn->view();
      }

      if (matchedIndicesSize > 0) {
        // Build a sequence [0..numProbeRows) and check which indices
        // appear in the semi-join result.
        auto probeRowSequence = cudf::sequence(
            numProbeRows,
            cudf::numeric_scalar<cudf::size_type>(
                0, true, stream, get_temp_mr()),
            cudf::numeric_scalar<cudf::size_type>(
                1, true, stream, get_temp_mr()),
            stream,
            get_temp_mr());

        auto matchedInBatch = cudf::contains(
            matchedIndicesView,
            probeRowSequence->view(),
            stream,
            get_temp_mr());

        matchFlags = cudf::binary_operation(
            matchFlags->view(),
            matchedInBatch->view(),
            cudf::binary_operator::BITWISE_OR,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            get_temp_mr());
      }
    }

    // Copy match flags into output memory resource since they go into the
    // output table passed downstream.
    auto outputMatchFlags = std::make_unique<cudf::column>(
        matchFlags->view(), stream, get_output_mr());

    // Assemble output: probe columns at their mapped positions + match column
    // at the last position.
    auto probeGatherView = probeTableView.select(probeColumnIndicesToGather_);
    auto gatheredProbe =
        std::make_unique<cudf::table>(probeGatherView, stream, get_output_mr());
    auto probeCols = gatheredProbe->release();

    auto numOutputColumns = outputType_->size();
    std::vector<std::unique_ptr<cudf::column>> outCols(numOutputColumns);
    for (size_t i = 0; i < probeColumnOutputIndices_.size(); ++i) {
      outCols[probeColumnOutputIndices_[i]] = std::move(probeCols[i]);
    }
    outCols[numOutputColumns - 1] = std::move(outputMatchFlags);

    auto result = std::make_unique<cudf::table>(std::move(outCols));
    input_.reset();

    if (result->num_rows() == 0) {
      return nullptr;
    }
    auto size = static_cast<vector_size_t>(result->num_rows());
    return std::make_shared<CudfVector>(
        operatorCtx_->pool(), outputType_, size, std::move(result), stream);
  }

  // For left/full join with filter: stream matches then emit mismatches.
  // Phase 1: the join may exceed one output batch (nested zones ⨝ 600M trips),
  // so process the probe input in cursor slices, returning one matched batch
  // per call and accumulating probeMatchedFlags_ across slices.
  // Phase 2 (cursor exhausted): emit unmatched probe rows once.
  if (isLeftOrFullJoin() && hasFilter_ && !buildEmpty_) {
    auto probeView = cudfInput->getTableView();
    auto numProbeRows = static_cast<cudf::size_type>(probeView.num_rows());
    while (spatialProbeCursor_ < numProbeRows) {
      cudf::size_type consumed = numProbeRows;
      auto result = joinWithBuildBatch(
          probeView,
          buildData_.value()->view(),
          stream,
          spatialProbeCursor_,
          &consumed);
      spatialProbeCursor_ = consumed;
      if (result.numRows > 0) {
        return std::make_shared<CudfVector>(
            operatorCtx_->pool(),
            outputType_,
            result.numRows,
            std::move(result.table),
            stream);
      }
      // This slice matched nothing; advance to the next slice.
    }

    // Emit unmatched probe rows with null build columns.
    auto mismatchResult = emitProbeMismatchRows(probeView, stream);
    input_.reset();
    probeMatchedFlags_.reset();
    spatialProbeCursor_ = 0;
    if (mismatchResult.numRows > 0) {
      return std::make_shared<CudfVector>(
          operatorCtx_->pool(),
          outputType_,
          mismatchResult.numRows,
          std::move(mismatchResult.table),
          stream);
    }
    return nullptr;
  }

  // Inner join: stream the probe input in cursor slices too, so a huge match
  // set (nested zones ⨝ 600M trips) never lands in one output batch.
  if (!buildEmpty_) {
    auto probeView = cudfInput->getTableView();
    auto numProbeRows = static_cast<cudf::size_type>(probeView.num_rows());
    while (spatialProbeCursor_ < numProbeRows) {
      cudf::size_type consumed = numProbeRows;
      auto result = joinWithBuildBatch(
          probeView,
          buildData_.value()->view(),
          stream,
          spatialProbeCursor_,
          &consumed);
      spatialProbeCursor_ = consumed;
      if (result.numRows > 0) {
        if (spatialProbeCursor_ >= numProbeRows) {
          input_.reset();
          spatialProbeCursor_ = 0;
        }
        return std::make_shared<CudfVector>(
            operatorCtx_->pool(),
            outputType_,
            result.numRows,
            std::move(result.table),
            stream);
      }
    }
    // Probe input fully consumed with no more matches.
    input_.reset();
    spatialProbeCursor_ = 0;
    return nullptr;
  }

  // Left/full join with empty build: emit all probe rows as mismatches.
  if (isLeftOrFullJoin() && buildEmpty_) {
    auto mismatchResult =
        emitProbeMismatchRows(cudfInput->getTableView(), stream);
    input_.reset();
    probeMatchedFlags_.reset();
    if (mismatchResult.numRows > 0) {
      return std::make_shared<CudfVector>(
          operatorCtx_->pool(),
          outputType_,
          mismatchResult.numRows,
          std::move(mismatchResult.table),
          stream);
    }
    return nullptr;
  }

  input_.reset();
  return nullptr;
}

// BridgeTranslator implementation
std::shared_ptr<const core::NestedLoopJoinNode> nestedLoopJoinFromSpatialJoin(
    const std::shared_ptr<const core::SpatialJoinNode>& spatialJoin) {
  VELOX_CHECK_NOT_NULL(spatialJoin);
  return std::make_shared<core::NestedLoopJoinNode>(
      spatialJoin->id(),
      spatialJoin->joinType(),
      spatialJoin->joinCondition(),
      spatialJoin->sources()[0],
      spatialJoin->sources()[1],
      spatialJoin->outputType());
}

std::unique_ptr<exec::Operator> CudfNestedLoopJoinBridgeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(node)) {
    return std::make_unique<CudfNestedLoopJoinProbe>(id, ctx, joinNode);
  }
  if (auto spatialJoin =
          std::dynamic_pointer_cast<const core::SpatialJoinNode>(node)) {
    SpatialEnvelopePrune prune;
    prune.probeGeometryName = spatialJoin->probeGeometry()->name();
    prune.buildGeometryName = spatialJoin->buildGeometry()->name();
    if (spatialJoin->radius().has_value()) {
      prune.buildRadiusName = spatialJoin->radius().value()->name();
      prune.predicate = SpatialPrunePredicate::kDistanceLE;
    } else if (spatialJoin->joinCondition()) {
      auto detected = trySpatialEnvelopePruneFromFilter(
          spatialJoin->joinCondition(),
          spatialJoin->sources()[0]->outputType(),
          spatialJoin->sources()[1]->outputType());
      if (detected.has_value()) {
        prune.predicate = detected->predicate;
        prune.withinPointOnBuild = detected->withinPointOnBuild;
        prune.probeIsWkb = detected->probeIsWkb;
        prune.buildIsWkb = detected->buildIsWkb;
        prune.filterIsCompound = detected->filterIsCompound;
        prune.residualFilter = detected->residualFilter;
        prune.buildConstantAabb = detected->buildConstantAabb;
        if (detected->probeGeometryName != detected->buildGeometryName) {
          prune.probeGeometryName = detected->probeGeometryName;
          prune.buildGeometryName = detected->buildGeometryName;
        }
      } else {
        auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(
            spatialJoin->joinCondition());
        bool isWithin = false;
        if (call) {
          auto const& name = call->name();
          if (name.size() >= 9) {
            auto tail = name.substr(name.size() - 9);
            isWithin = true;
            for (size_t i = 0; i < 9; ++i) {
              if (std::tolower(static_cast<unsigned char>(tail[i])) !=
                  "st_within"[i]) {
                isWithin = false;
                break;
              }
            }
          }
        }
        prune.predicate = isWithin ? SpatialPrunePredicate::kWithin
                                   : SpatialPrunePredicate::kIntersects;
      }
    } else {
      prune.predicate = SpatialPrunePredicate::kIntersects;
    }
    return std::make_unique<CudfNestedLoopJoinProbe>(
        id, ctx, nestedLoopJoinFromSpatialJoin(spatialJoin), std::move(prune));
  }
  return nullptr;
}

std::unique_ptr<exec::JoinBridge>
CudfNestedLoopJoinBridgeTranslator::toJoinBridge(
    const core::PlanNodePtr& node) {
  if (std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(node) ||
      std::dynamic_pointer_cast<const core::SpatialJoinNode>(node)) {
    return std::make_unique<CudfNestedLoopJoinBridge>();
  }
  return nullptr;
}

exec::OperatorSupplier CudfNestedLoopJoinBridgeTranslator::toOperatorSupplier(
    const core::PlanNodePtr& node) {
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(node)) {
    return [joinNode](int32_t operatorId, exec::DriverCtx* ctx) {
      return std::make_unique<CudfNestedLoopJoinBuild>(
          operatorId, ctx, joinNode);
    };
  }
  if (auto spatialJoin =
          std::dynamic_pointer_cast<const core::SpatialJoinNode>(node)) {
    auto joinNode = nestedLoopJoinFromSpatialJoin(spatialJoin);
    return [joinNode](int32_t operatorId, exec::DriverCtx* ctx) {
      return std::make_unique<CudfNestedLoopJoinBuild>(
          operatorId, ctx, joinNode);
    };
  }
  return nullptr;
}

} // namespace facebook::velox::cudf_velox
