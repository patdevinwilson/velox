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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/types/GeometryType.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

/// Build a Velox GeometrySerde POINT blob: tag(0) + x + y.
std::string makePointBlob(double x, double y) {
  std::string blob(17, '\0');
  blob[0] = static_cast<char>(0); // GeometrySerializationType::POINT
  std::memcpy(blob.data() + 1, &x, sizeof(double));
  std::memcpy(blob.data() + 9, &y, sizeof(double));
  return blob;
}

/// Little-endian WKB Point (ISO type 1): endian + type + x + y.
std::string makeWkbPoint(double x, double y) {
  std::string wkb(21, '\0');
  wkb[0] = 1; // NDR
  uint32_t type = 1;
  std::memcpy(wkb.data() + 1, &type, sizeof(type));
  std::memcpy(wkb.data() + 5, &x, sizeof(double));
  std::memcpy(wkb.data() + 13, &y, sizeof(double));
  return wkb;
}

class CudfGeospatialTest : public testing::Test,
                           public facebook::velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    if (!isRegisteredVectorSerde()) {
      serializer::presto::PrestoVectorSerde::registerVectorSerde();
    }
    functions::prestosql::registerAllScalarFunctions();
    parse::registerTypeResolver();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
    functions::prestosql::registerAllScalarFunctions(
        cudf_velox::CudfConfig::getInstance().functionNamePrefix);
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
  }

  VectorPtr makeGeometryPoints(
      const std::vector<double>& xs,
      const std::vector<double>& ys) {
    VELOX_CHECK_EQ(xs.size(), ys.size());
    std::vector<std::string> blobs;
    blobs.reserve(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
      blobs.push_back(makePointBlob(xs[i], ys[i]));
    }
    // StringViews alias blobs only during FlatVector construction (copied in).
    std::vector<StringView> views;
    views.reserve(blobs.size());
    for (const auto& blob : blobs) {
      views.emplace_back(blob);
    }
    return makeFlatVector<StringView>(views, GEOMETRY());
  }

  static double haversineKm(
      double lat1,
      double lon1,
      double lat2,
      double lon2) {
    constexpr double kEarthRadiusKm = 6371.0088;
    constexpr double kDeg2Rad = M_PI / 180.0;
    double dLat = (lat2 - lat1) * kDeg2Rad;
    double dLon = (lon2 - lon1) * kDeg2Rad;
    double a = std::sin(dLat / 2) * std::sin(dLat / 2) +
        std::cos(lat1 * kDeg2Rad) * std::cos(lat2 * kDeg2Rad) *
            std::sin(dLon / 2) * std::sin(dLon / 2);
    return 2.0 * kEarthRadiusKm * std::asin(std::sqrt(a));
  }
};

TEST_F(CudfGeospatialTest, greatCircleDistanceSamePoint) {
  auto lat = makeFlatVector<double>({0.0, 40.7128, -33.8688});
  auto lon = makeFlatVector<double>({0.0, -74.0060, 151.2093});
  auto data =
      makeRowVector({"lat1", "lon1", "lat2", "lon2"}, {lat, lon, lat, lon});

  auto plan = PlanBuilder()
                  .values({data})
                  .project({"great_circle_distance(lat1, lon1, lat2, lon2)"})
                  .planNode();

  auto expected =
      makeRowVector({"p0"}, {makeFlatVector<double>({0.0, 0.0, 0.0})});

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfGeospatialTest, greatCircleDistanceKnownPairs) {
  double nyLat = 40.7128, nyLon = -74.0060;
  double laLat = 33.9425, laLon = -118.4081;
  double londonLat = 51.5074, londonLon = -0.1278;
  double tokyoLat = 35.6762, tokyoLon = 139.6503;

  auto lat1 = makeFlatVector<double>({nyLat, nyLat, londonLat});
  auto lon1 = makeFlatVector<double>({nyLon, nyLon, londonLon});
  auto lat2 = makeFlatVector<double>({laLat, londonLat, tokyoLat});
  auto lon2 = makeFlatVector<double>({laLon, londonLon, tokyoLon});
  auto data =
      makeRowVector({"lat1", "lon1", "lat2", "lon2"}, {lat1, lon1, lat2, lon2});

  auto plan = PlanBuilder()
                  .values({data})
                  .project({"great_circle_distance(lat1, lon1, lat2, lon2)"})
                  .planNode();

  // GPU uses Vincenty (BingTile); allow a few km vs haversine reference.
  double nyToLa = haversineKm(nyLat, nyLon, laLat, laLon);
  double nyToLondon = haversineKm(nyLat, nyLon, londonLat, londonLon);
  double londonToTokyo = haversineKm(londonLat, londonLon, tokyoLat, tokyoLon);

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  auto resultVec = result->childAt(0)->asFlatVector<double>();

  ASSERT_EQ(resultVec->size(), 3);
  EXPECT_NEAR(resultVec->valueAt(0), nyToLa, 5.0);
  EXPECT_NEAR(resultVec->valueAt(1), nyToLondon, 5.0);
  EXPECT_NEAR(resultVec->valueAt(2), londonToTokyo, 5.0);
}

TEST_F(CudfGeospatialTest, stXStYFromPoint) {
  auto data = makeRowVector(
      {"g"}, {makeGeometryPoints({1.5, -2.0, 10.0}, {3.25, 4.0, -5.5})});

  auto plan = PlanBuilder()
                  .values({data})
                  .project({"ST_X(g)", "ST_Y(g)"})
                  .planNode();

  auto expected = makeRowVector(
      {"p0", "p1"},
      {makeFlatVector<double>({1.5, -2.0, 10.0}),
       makeFlatVector<double>({3.25, 4.0, -5.5})});

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfGeospatialTest, stPointRoundTrip) {
  auto data = makeRowVector(
      {"x", "y"},
      {makeFlatVector<double>({1.5, -2.0, 10.0}),
       makeFlatVector<double>({3.25, 4.0, -5.5})});

  auto plan = PlanBuilder()
                  .values({data})
                  .project({"ST_X(ST_Point(x, y))", "ST_Y(ST_Point(x, y))"})
                  .planNode();

  auto expected = makeRowVector(
      {"p0", "p1"},
      {makeFlatVector<double>({1.5, -2.0, 10.0}),
       makeFlatVector<double>({3.25, 4.0, -5.5})});

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfGeospatialTest, stDistancePointPoint) {
  // Matches GeometryFunctionsTest Euclidean point-point case:
  // POINT(50 100) to POINT(150 150) => ~111.80339887498948
  auto g1 = makeGeometryPoints({50.0, 0.0}, {100.0, 0.0});
  auto g2 = makeGeometryPoints({150.0, 0.0}, {150.0, 0.0});
  auto data = makeRowVector({"a", "b"}, {g1, g2});

  auto plan = PlanBuilder()
                  .values({data})
                  .project({"ST_Distance(a, b)"})
                  .planNode();

  auto expected = makeRowVector(
      {"p0"},
      {makeFlatVector<double>({111.80339887498948, 0.0})});

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfGeospatialTest, stGeomFromBinaryAndQ1Shape) {
  // SpatialBench Q1 shape: ST_X/ST_Y/ST_Distance over ST_GeomFromBinary,
  // with a constant center from ST_GeometryFromText (constant-folded).
  std::vector<std::string> owned = {
      makeWkbPoint(1.5, 3.25),
      makeWkbPoint(-2.0, 4.0),
      makeWkbPoint(-111.7610, 34.8697)};
  std::vector<StringView> views;
  views.reserve(owned.size());
  for (const auto& s : owned) {
    views.emplace_back(s);
  }
  auto data = makeRowVector(
      {"pickup"}, {makeFlatVector<StringView>(views, VARBINARY())});

  auto plan = PlanBuilder()
                  .values({data})
                  .project(
                      {"ST_X(ST_GeomFromBinary(pickup))",
                       "ST_Y(ST_GeomFromBinary(pickup))",
                       "ST_Distance(ST_GeomFromBinary(pickup), ST_GeometryFromText('POINT (-111.7610 34.8697)'))"})
                  .planNode();

  double d0 = std::hypot(1.5 - (-111.7610), 3.25 - 34.8697);
  double d1 = std::hypot(-2.0 - (-111.7610), 4.0 - 34.8697);
  auto expected = makeRowVector(
      {"p0", "p1", "p2"},
      {makeFlatVector<double>({1.5, -2.0, -111.7610}),
       makeFlatVector<double>({3.25, 4.0, 34.8697}),
       makeFlatVector<double>({d0, d1, 0.0})});

  AssertQueryBuilder(plan).assertResults(expected);
}

} // namespace

