# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include_guard(GLOBAL)

# NOTE: cuSpatial's last release (branch-25.08, cut 2025-07-28) targets cuDF
# 25.08 and its `main` branch has been inactive since 2025-04-09, so there is
# no cuSpatial release that is API/ABI-compatible with the cuDF commit pinned
# above (release/26.08, 2026-07-27). Linking libcuspatial's cuDF-column-based
# API (distance.hpp et al.) against this newer cuDF would not build.
#
# Instead we vendor only cuSpatial's header-only algorithm API (distance.cuh
# and friends), which is explicitly documented as independent of libcudf --
# it is templated on iterators and depends only on Thrust/RMM/CUDA. We fetch
# the cuSpatial source tree for its headers only; we never add_subdirectory
# into it (it has no top-level CMakeLists.txt, only cpp/CMakeLists.txt, so
# FetchContent_MakeAvailable populates it without trying to configure it as a
# CMake subproject) and we never include any of its *.hpp column-API headers.

# cuspatial commit 05f4a858a from 2025-07-28 (branch-25.08 tip; latest release,
# see note above on why a newer pin is not available)
set(VELOX_cuspatial_VERSION 25.08 CACHE STRING "cuspatial version")
set(VELOX_cuspatial_COMMIT 05f4a858aed58c8cae7178c02fcfb31a5215b26e)
set(
  VELOX_cuspatial_BUILD_SHA256_CHECKSUM
  2910b89d2d7b45beeb1bf72eb7ef22937ad27758a8b6f80ce1401b73be3010e1
)
set(
  VELOX_cuspatial_SOURCE_URL
  "https://github.com/rapidsai/cuspatial/archive/${VELOX_cuspatial_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(cuspatial)

FetchContent_Declare(
  cuspatial
  URL ${VELOX_cuspatial_SOURCE_URL}
  URL_HASH ${VELOX_cuspatial_BUILD_SHA256_CHECKSUM}
  UPDATE_DISCONNECTED 1
)

# No SOURCE_SUBDIR/add_subdirectory: this only downloads+extracts the source
# tree so we can point an include path at cpp/include. cuSpatial has no
# top-level CMakeLists.txt, so MakeAvailable will not try to configure it.
FetchContent_MakeAvailable(cuspatial)

# cuSpatial's header-only range/distance algorithms pull in
# github.com/harrism/ranger (a tiny header-only grid-stride-range utility,
# Apache-2.0) for its detail namespace helpers. It has no build system either;
# we only need the header. cuSpatial itself pins this to the tip of `main`
# (see cpp/cmake/thirdparty/get_ranger.cmake); main has been a single stable
# commit since 2023-06-14, so we pin that exact commit for reproducibility.
set(VELOX_ranger_COMMIT 9604a9ea8c90a601b715a582ed8163d9526858a9)
set(
  VELOX_ranger_BUILD_SHA256_CHECKSUM
  8f55b523cbb1730930fe1bd1d6fa329338b2cf2bc00ae883443487c23a375ea8
)
set(
  VELOX_ranger_SOURCE_URL
  "https://github.com/harrism/ranger/archive/${VELOX_ranger_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(ranger)

FetchContent_Declare(
  ranger
  URL ${VELOX_ranger_SOURCE_URL}
  URL_HASH ${VELOX_ranger_BUILD_SHA256_CHECKSUM}
  UPDATE_DISCONNECTED 1
)
FetchContent_MakeAvailable(ranger)

# cuSpatial 25.08 still includes RMM headers at their pre-26.x path
# (rmm/mr/device/*.hpp); modern RMM (pinned above for cuDF 26.08) moved these
# up to rmm/mr/*.hpp. Rather than patch cuSpatial's vendored source, ship a
# tiny compatibility shim directory that forwards the handful of old paths
# cuSpatial's distance/*.cuh headers transitively include.
set(VELOX_CUSPATIAL_RMM_COMPAT_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuspatial_rmm_compat_shim")
file(
  WRITE
  "${VELOX_CUSPATIAL_RMM_COMPAT_DIR}/rmm/mr/device/per_device_resource.hpp"
  "// Compat shim: cuSpatial 25.08 includes the pre-26.x RMM header path\n"
  "// rmm/mr/device/per_device_resource.hpp; modern RMM moved this to\n"
  "// rmm/mr/per_device_resource.hpp.\n"
  "#pragma once\n"
  "#include <rmm/mr/per_device_resource.hpp>\n"
)

# cuSpatial 25.08's cpp/include/cuspatial/detail/distance/point_polygon_distance.cuh
# calls `point_polygon_intersects(multipoints, multipolygons, stream)` unqualified
# from within `namespace cuspatial { ... }`, but that helper is actually defined in
# the nested `cuspatial::detail` namespace (distance_utils.cuh). Because the call's
# arguments are cuspatial::multipoint_range/multipolygon_range -- themselves declared
# in `namespace cuspatial`, not `cuspatial::detail` -- neither ordinary unqualified
# lookup nor ADL finds `cuspatial::detail::point_polygon_intersects` from that call
# site: this is a genuine missing-qualification bug in that release, not merely a
# strict-vs-lax two-phase-lookup difference (`-fpermissive` cannot fix it, since the
# name truly doesn't exist in scope; it only forwards to the host compiler, not
# nvcc's own frontend anyway). Fix it by pulling the `detail` overload into the
# `cuspatial` namespace via a `using`-declaration, in a shim header that must be
# `#include`d (from consuming code, e.g. GeometryKernels.cu) before
# <cuspatial/distance.cuh>, so the using-declaration is visible by the time
# point_polygon_distance.cuh's template is parsed.
file(
  WRITE
  "${VELOX_CUSPATIAL_RMM_COMPAT_DIR}/cuspatial_compat_fixups.cuh"
  "// See CMake/resolve_dependency_modules/cuspatial.cmake for why this shim\n"
  "// exists. Must be included before <cuspatial/distance.cuh>.\n"
  "#pragma once\n"
  "// Several cuSpatial 25.08 headers call thrust::next/thrust::prev/\n"
  "// thrust::get/thrust::distance without including their defining headers\n"
  "// (thrust/advance.h, thrust/pair.h, thrust/distance.h respectively);\n"
  "// cuSpatial relied on some other header transitively pulling these in,\n"
  "// which current CCCL/Thrust no longer does (header hygiene tightened).\n"
  "// Include them explicitly.\n"
  "#include <thrust/advance.h>\n"
  "#include <thrust/distance.h>\n"
  "#include <thrust/pair.h>\n"
  "#include <thrust/tuple.h>\n"
  "#include <cuspatial/detail/distance/distance_utils.cuh>\n"
  "namespace cuspatial {\n"
  "using detail::point_polygon_intersects;\n"
  "}\n"
)

if(NOT TARGET cuspatial::headers)
  add_library(cuspatial_headers INTERFACE)
  add_library(cuspatial::headers ALIAS cuspatial_headers)
  target_include_directories(
    cuspatial_headers
    SYSTEM
    INTERFACE
      ${VELOX_CUSPATIAL_RMM_COMPAT_DIR}
      ${cuspatial_SOURCE_DIR}/cpp/include
      ${ranger_SOURCE_DIR}/include
  )
  target_compile_options(
    cuspatial_headers
    INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
endif()
