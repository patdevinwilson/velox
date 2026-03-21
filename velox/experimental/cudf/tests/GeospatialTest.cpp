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
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

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
  auto data = makeRowVector({"lat1", "lon1", "lat2", "lon2"},
      {lat, lon, lat, lon});

  auto plan = PlanBuilder()
                  .values({data})
                  .project({"great_circle_distance(lat1, lon1, lat2, lon2)"})
                  .planNode();

  auto expected = makeRowVector(
      {"p0"},
      {makeFlatVector<double>({0.0, 0.0, 0.0})});

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
  auto data = makeRowVector({"lat1", "lon1", "lat2", "lon2"},
      {lat1, lon1, lat2, lon2});

  auto plan = PlanBuilder()
                  .values({data})
                  .project({"great_circle_distance(lat1, lon1, lat2, lon2)"})
                  .planNode();

  double nyToLa = haversineKm(nyLat, nyLon, laLat, laLon);
  double nyToLondon = haversineKm(nyLat, nyLon, londonLat, londonLon);
  double londonToTokyo = haversineKm(londonLat, londonLon, tokyoLat, tokyoLon);

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  auto resultVec = result->childAt(0)->asFlatVector<double>();

  ASSERT_EQ(resultVec->size(), 3);
  EXPECT_NEAR(resultVec->valueAt(0), nyToLa, 0.01);
  EXPECT_NEAR(resultVec->valueAt(1), nyToLondon, 0.01);
  EXPECT_NEAR(resultVec->valueAt(2), londonToTokyo, 0.01);

  EXPECT_NEAR(nyToLa, 3955.0, 50.0);
  EXPECT_NEAR(nyToLondon, 5570.0, 50.0);
  EXPECT_NEAR(londonToTokyo, 9560.0, 50.0);
}

} // namespace
