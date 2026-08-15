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

/// Little-endian WKB Polygon (ISO type 3), single closed exterior ring.
std::string makeWkbPolygon(const std::vector<std::pair<double, double>>& ring) {
  VELOX_CHECK_GE(ring.size(), 4);
  std::string wkb;
  wkb.resize(1 + 4 + 4 + 4 + ring.size() * 16);
  wkb[0] = 1; // NDR
  uint32_t type = 3;
  uint32_t numRings = 1;
  uint32_t numPoints = static_cast<uint32_t>(ring.size());
  std::memcpy(wkb.data() + 1, &type, sizeof(type));
  std::memcpy(wkb.data() + 5, &numRings, sizeof(numRings));
  std::memcpy(wkb.data() + 9, &numPoints, sizeof(numPoints));
  for (size_t i = 0; i < ring.size(); ++i) {
    std::memcpy(
        wkb.data() + 13 + i * 16, &ring[i].first, sizeof(double));
    std::memcpy(
        wkb.data() + 13 + i * 16 + 8, &ring[i].second, sizeof(double));
  }
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

TEST_F(CudfGeospatialTest, stDistancePointConstantPolygon) {
  // SpatialBench Q3 shape: point vs constant axis-aligned POLYGON.
  // POLYGON((-111.9060 34.7347, -111.6160 34.7347, -111.6160 35.0047,
  //          -111.9060 35.0047, -111.9060 34.7347))
  std::vector<std::string> owned = {
      makeWkbPoint(-111.0, 34.8), // outside to the east
      makeWkbPoint(-111.76, 34.87), // inside
      makeWkbPoint(-111.7610, 34.7000), // south of box
  };
  std::vector<StringView> views;
  for (const auto& s : owned) {
    views.emplace_back(s);
  }
  auto data = makeRowVector(
      {"pickup"}, {makeFlatVector<StringView>(views, VARBINARY())});

  const char* poly =
      "POLYGON((-111.9060 34.7347, -111.6160 34.7347, -111.6160 35.0047, -111.9060 35.0047, -111.9060 34.7347))";
  std::string distExpr =
      "ST_Distance(ST_GeomFromBinary(pickup), ST_GeometryFromText('" +
      std::string(poly) + "'))";
  auto plan = PlanBuilder()
                  .values({data})
                  .project({distExpr})
                  .planNode();

  // East of xmax=-111.6160 at y=34.8 (inside y-range): dx=0.616
  double dEast = -111.0 - (-111.6160);
  // Inside → 0
  // South of ymin=34.7347 at x=-111.761 (inside x-range): dy=0.0347
  double dSouth = 34.7347 - 34.7000;

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  auto dists = result->childAt(0)->asFlatVector<double>();
  ASSERT_EQ(dists->size(), 3);
  EXPECT_NEAR(dists->valueAt(0), dEast, 1e-9);
  EXPECT_NEAR(dists->valueAt(1), 0.0, 1e-12);
  EXPECT_NEAR(dists->valueAt(2), dSouth, 1e-9);
}

TEST_F(CudfGeospatialTest, stDistancePointPolygonColumnQ8Shape) {
  // SpatialBench Q8 shape: ST_Distance(ST_GeomFromBinary(pickup),
  // ST_GeomFromBinary(building_boundary)) — column POINT vs column POLYGON.
  auto poly = makeWkbPolygon({
      {-111.9060, 34.7347},
      {-111.6160, 34.7347},
      {-111.6160, 35.0047},
      {-111.9060, 35.0047},
      {-111.9060, 34.7347},
  });
  std::vector<std::string> pickups = {
      makeWkbPoint(-111.0, 34.8), // east of box
      makeWkbPoint(-111.76, 34.87), // inside
      makeWkbPoint(-111.7610, 34.7000), // south of box
  };
  std::vector<std::string> boundaries = {poly, poly, poly};
  std::vector<StringView> pickupViews;
  std::vector<StringView> boundaryViews;
  for (const auto& s : pickups) {
    pickupViews.emplace_back(s);
  }
  for (const auto& s : boundaries) {
    boundaryViews.emplace_back(s);
  }
  auto data = makeRowVector(
      {"pickup", "b_boundary"},
      {makeFlatVector<StringView>(pickupViews, VARBINARY()),
       makeFlatVector<StringView>(boundaryViews, VARBINARY())});

  auto plan =
      PlanBuilder()
          .values({data})
          .project(
              {"ST_Distance(ST_GeomFromBinary(pickup), ST_GeomFromBinary(b_boundary))"})
          .planNode();

  double dEast = -111.0 - (-111.6160);
  double dSouth = 34.7347 - 34.7000;

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  auto dists = result->childAt(0)->asFlatVector<double>();
  ASSERT_EQ(dists->size(), 3);
  EXPECT_NEAR(dists->valueAt(0), dEast, 1e-9);
  EXPECT_NEAR(dists->valueAt(1), 0.0, 1e-12);
  EXPECT_NEAR(dists->valueAt(2), dSouth, 1e-9);
}

TEST_F(CudfGeospatialTest, stLengthLineStringQ7Shape) {
  // SpatialBench Q7:
  // ST_Length(ST_LineString(ARRAY[ST_GeomFromBinary(pickup),
  //                                ST_GeomFromBinary(dropoff)])) / 0.000009
  std::vector<std::string> pickups = {
      makeWkbPoint(0.0, 0.0),
      makeWkbPoint(-111.7610, 34.8697),
      makeWkbPoint(1.0, 1.0),
  };
  std::vector<std::string> dropoffs = {
      makeWkbPoint(3.0, 4.0), // length 5
      makeWkbPoint(-111.6160, 34.8697), // east 0.145 degrees
      makeWkbPoint(1.0, 2.0), // length 1
  };
  std::vector<StringView> pickupViews;
  std::vector<StringView> dropoffViews;
  for (const auto& s : pickups) {
    pickupViews.emplace_back(s);
  }
  for (const auto& s : dropoffs) {
    dropoffViews.emplace_back(s);
  }
  auto data = makeRowVector(
      {"pickup", "dropoff"},
      {makeFlatVector<StringView>(pickupViews, VARBINARY()),
       makeFlatVector<StringView>(dropoffViews, VARBINARY())});

  auto plan =
      PlanBuilder()
          .values({data})
          .project(
              {"ST_Length(ST_LineString(ARRAY[ST_GeomFromBinary(pickup), ST_GeomFromBinary(dropoff)])) / 0.000009"})
          .planNode();

  double d0 = 5.0 / 0.000009;
  double d1 = 0.145 / 0.000009;
  double d2 = 1.0 / 0.000009;

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  auto dists = result->childAt(0)->asFlatVector<double>();
  ASSERT_EQ(dists->size(), 3);
  EXPECT_NEAR(dists->valueAt(0), d0, 1e-6);
  EXPECT_NEAR(dists->valueAt(1), d1, 1e-6);
  EXPECT_NEAR(dists->valueAt(2), d2, 1e-6);
}

TEST_F(CudfGeospatialTest, stLineStringAllowsDuplicatePoints) {
  // Sedona ST_MakeLine / SpatialBench Q7 parity: consecutive duplicate points
  // are valid; ST_Length of the zero-length segment is 0.
  std::vector<std::string> pickups = {makeWkbPoint(1.0, 2.0)};
  std::vector<std::string> dropoffs = {makeWkbPoint(1.0, 2.0)};
  std::vector<StringView> pickupViews;
  std::vector<StringView> dropoffViews;
  pickupViews.emplace_back(pickups[0]);
  dropoffViews.emplace_back(dropoffs[0]);
  auto data = makeRowVector(
      {"pickup", "dropoff"},
      {makeFlatVector<StringView>(pickupViews, VARBINARY()),
       makeFlatVector<StringView>(dropoffViews, VARBINARY())});

  auto plan =
      PlanBuilder()
          .values({data})
          .project(
              {"ST_Length(ST_LineString(ARRAY[ST_GeomFromBinary(pickup), ST_GeomFromBinary(dropoff)]))"})
          .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  auto lengths = result->childAt(0)->asFlatVector<double>();
  ASSERT_EQ(lengths->size(), 1);
  EXPECT_NEAR(lengths->valueAt(0), 0.0, 1e-12);
}

TEST_F(CudfGeospatialTest, stWithinPointConstantPolygon) {
  // Q2/Q4 shape: ST_Within / ST_Intersects point vs constant polygon.
  std::vector<std::string> owned = {
      makeWkbPoint(-111.0, 34.8), // outside
      makeWkbPoint(-111.76, 34.87), // inside
      makeWkbPoint(-111.7610, 34.7000), // outside south
  };
  std::vector<StringView> views;
  for (const auto& s : owned) {
    views.emplace_back(s);
  }
  auto data = makeRowVector(
      {"pickup"}, {makeFlatVector<StringView>(views, VARBINARY())});

  const char* poly =
      "POLYGON((-111.9060 34.7347, -111.6160 34.7347, -111.6160 35.0047, -111.9060 35.0047, -111.9060 34.7347))";
  std::string withinExpr =
      "ST_Within(ST_GeomFromBinary(pickup), ST_GeometryFromText('" +
      std::string(poly) + "'))";
  std::string intersectsExpr =
      "ST_Intersects(ST_GeomFromBinary(pickup), ST_GeometryFromText('" +
      std::string(poly) + "'))";
  std::string containsExpr =
      "ST_Contains(ST_GeometryFromText('" + std::string(poly) +
      "'), ST_GeomFromBinary(pickup))";

  auto plan = PlanBuilder()
                  .values({data})
                  .project({withinExpr, intersectsExpr, containsExpr})
                  .planNode();

  auto expected = makeRowVector(
      {"p0", "p1", "p2"},
      {makeFlatVector<bool>({false, true, false}),
       makeFlatVector<bool>({false, true, false}),
       makeFlatVector<bool>({false, true, false})});

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfGeospatialTest, stWithinPointPolygonColumn) {
  auto poly = makeWkbPolygon({
      {-111.9060, 34.7347},
      {-111.6160, 34.7347},
      {-111.6160, 35.0047},
      {-111.9060, 35.0047},
      {-111.9060, 34.7347},
  });
  std::vector<std::string> pickups = {
      makeWkbPoint(-111.0, 34.8),
      makeWkbPoint(-111.76, 34.87),
      makeWkbPoint(-111.7610, 34.7000),
  };
  std::vector<std::string> boundaries = {poly, poly, poly};
  std::vector<StringView> pickupViews;
  std::vector<StringView> boundaryViews;
  for (const auto& s : pickups) {
    pickupViews.emplace_back(s);
  }
  for (const auto& s : boundaries) {
    boundaryViews.emplace_back(s);
  }
  auto data = makeRowVector(
      {"pickup", "z_boundary"},
      {makeFlatVector<StringView>(pickupViews, VARBINARY()),
       makeFlatVector<StringView>(boundaryViews, VARBINARY())});

  auto plan = PlanBuilder()
                  .values({data})
                  .project(
                      {"ST_Within(ST_GeomFromBinary(pickup), ST_GeomFromBinary(z_boundary))"})
                  .planNode();

  auto expected = makeRowVector(
      {"p0"}, {makeFlatVector<bool>({false, true, false})});

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfGeospatialTest, stIntersectsPolygonPolygonQ6Shape) {
  // Q6: axis-aligned bbox intersects zone polygon.
  auto zone = makeWkbPolygon({
      {-111.9060, 34.7347},
      {-111.6160, 34.7347},
      {-111.6160, 35.0047},
      {-111.9060, 35.0047},
      {-111.9060, 34.7347},
  });
  auto farZone = makeWkbPolygon({
      {-110.0, 33.0},
      {-109.0, 33.0},
      {-109.0, 33.5},
      {-110.0, 33.5},
      {-110.0, 33.0},
  });
  std::vector<std::string> zones = {zone, farZone};
  std::vector<StringView> zoneViews;
  for (const auto& s : zones) {
    zoneViews.emplace_back(s);
  }
  auto data = makeRowVector(
      {"z_boundary"}, {makeFlatVector<StringView>(zoneViews, VARBINARY())});

  // Overlaps first zone; misses second.
  const char* bbox =
      "POLYGON((-112.2110 34.4197, -111.3110 34.4197, -111.3110 35.3197, -112.2110 35.3197, -112.2110 34.4197))";
  std::string expr =
      "ST_Intersects(ST_GeometryFromText('" + std::string(bbox) +
      "'), ST_GeomFromBinary(z_boundary))";
  auto plan = PlanBuilder().values({data}).project({expr}).planNode();

  auto expected =
      makeRowVector({"p0"}, {makeFlatVector<bool>({true, false})});

  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfGeospatialTest, stAreaPolygonQ9Shape) {
  // SpatialBench Q9: ST_Area(ST_GeomFromBinary(b_boundary)).
  // Unit square → 1; 2×3 rectangle → 6; point → 0 (GEOS non-areal).
  auto unitSq = makeWkbPolygon({
      {0.0, 0.0},
      {1.0, 0.0},
      {1.0, 1.0},
      {0.0, 1.0},
      {0.0, 0.0},
  });
  auto rect = makeWkbPolygon({
      {0.0, 0.0},
      {2.0, 0.0},
      {2.0, 3.0},
      {0.0, 3.0},
      {0.0, 0.0},
  });
  std::vector<std::string> wkbs = {unitSq, rect, makeWkbPoint(1.0, 2.0)};
  std::vector<StringView> views;
  for (const auto& s : wkbs) {
    views.emplace_back(s);
  }
  auto data = makeRowVector(
      {"geom"}, {makeFlatVector<StringView>(views, VARBINARY())});

  auto plan = PlanBuilder()
                  .values({data})
                  .project({"ST_Area(ST_GeomFromBinary(geom))"})
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  auto areas = result->childAt(0)->asFlatVector<double>();
  ASSERT_EQ(areas->size(), 3);
  EXPECT_NEAR(areas->valueAt(0), 1.0, 1e-9);
  EXPECT_NEAR(areas->valueAt(1), 6.0, 1e-9);
  EXPECT_NEAR(areas->valueAt(2), 0.0, 1e-12);
}

TEST_F(CudfGeospatialTest, stIntersectionAreaQ9Shape) {
  // Q9 core: ST_Area(ST_Intersection(a, b)) for overlapping unit squares.
  // [0,1]x[0,1] ∩ [0.5,1.5]x[0.5,1.5] → 0.5×0.5 = 0.25
  // Disjoint → 0.
  auto a = makeWkbPolygon({
      {0.0, 0.0},
      {1.0, 0.0},
      {1.0, 1.0},
      {0.0, 1.0},
      {0.0, 0.0},
  });
  auto bOverlap = makeWkbPolygon({
      {0.5, 0.5},
      {1.5, 0.5},
      {1.5, 1.5},
      {0.5, 1.5},
      {0.5, 0.5},
  });
  auto bFar = makeWkbPolygon({
      {10.0, 10.0},
      {11.0, 10.0},
      {11.0, 11.0},
      {10.0, 11.0},
      {10.0, 10.0},
  });
  std::vector<std::string> lefts = {a, a};
  std::vector<std::string> rights = {bOverlap, bFar};
  std::vector<StringView> lv, rv;
  for (const auto& s : lefts) {
    lv.emplace_back(s);
  }
  for (const auto& s : rights) {
    rv.emplace_back(s);
  }
  auto data = makeRowVector(
      {"a", "b"},
      {makeFlatVector<StringView>(lv, VARBINARY()),
       makeFlatVector<StringView>(rv, VARBINARY())});

  auto plan = PlanBuilder()
                  .values({data})
                  .project(
                      {"ST_Area(ST_Intersection(ST_GeomFromBinary(a), ST_GeomFromBinary(b)))"})
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  auto areas = result->childAt(0)->asFlatVector<double>();
  ASSERT_EQ(areas->size(), 2);
  EXPECT_NEAR(areas->valueAt(0), 0.25, 1e-6);
  EXPECT_NEAR(areas->valueAt(1), 0.0, 1e-9);
}

TEST_F(CudfGeospatialTest, stConvexHullAreaQ5Shape) {
  // Q5 core: ST_Area(ST_ConvexHull(geometry_union(ARRAY[points]))).
  // Unit-square corners + interior → hull area 1.
  // Two points → line → area 0.
  auto p00 = makeWkbPoint(0.0, 0.0);
  auto p10 = makeWkbPoint(1.0, 0.0);
  auto p11 = makeWkbPoint(1.0, 1.0);
  auto p01 = makeWkbPoint(0.0, 1.0);
  auto pMid = makeWkbPoint(0.5, 0.5);

  // Build via ARRAY[] in projection (matches SpatialBench Q5 shape).
  auto square = makeRowVector(
      {"p0", "p1", "p2", "p3", "p4"},
      {makeFlatVector<StringView>({StringView(p00)}, VARBINARY()),
       makeFlatVector<StringView>({StringView(p10)}, VARBINARY()),
       makeFlatVector<StringView>({StringView(p11)}, VARBINARY()),
       makeFlatVector<StringView>({StringView(p01)}, VARBINARY()),
       makeFlatVector<StringView>({StringView(pMid)}, VARBINARY())});
  auto planSquare =
      PlanBuilder()
          .values({square})
          .project(
              {"ST_Area(ST_ConvexHull(geometry_union(ARRAY["
               "ST_GeomFromBinary(p0), ST_GeomFromBinary(p1), "
               "ST_GeomFromBinary(p2), ST_GeomFromBinary(p3), "
               "ST_GeomFromBinary(p4)])))"})
          .planNode();
  auto resSquare = AssertQueryBuilder(planSquare).copyResults(pool());
  auto areaSquare = resSquare->childAt(0)->asFlatVector<double>();
  ASSERT_EQ(areaSquare->size(), 1);
  EXPECT_NEAR(areaSquare->valueAt(0), 1.0, 1e-6);

  auto line = makeRowVector(
      {"p0", "p1"},
      {makeFlatVector<StringView>({StringView(p00)}, VARBINARY()),
       makeFlatVector<StringView>({StringView(p11)}, VARBINARY())});
  auto planLine =
      PlanBuilder()
          .values({line})
          .project(
              {"ST_Area(ST_ConvexHull(geometry_union(ARRAY["
               "ST_GeomFromBinary(p0), ST_GeomFromBinary(p1)])))"})
          .planNode();
  auto resLine = AssertQueryBuilder(planLine).copyResults(pool());
  auto areaLine = resLine->childAt(0)->asFlatVector<double>();
  ASSERT_EQ(areaLine->size(), 1);
  EXPECT_NEAR(areaLine->valueAt(0), 0.0, 1e-12);
}

} // namespace

