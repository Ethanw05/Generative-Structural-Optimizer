# Every third-party dependency, pinned by version and SHA-256 and fetched at configure
# time. Nothing is taken from the system. Each library is added in the phase whose code
# first uses it (see docs/port-plan.md, section 6), so nothing unused is built.
#
# All are permissive licences usable in a closed-source commercial product:
#   Eigen 5.0.1        MPL-2.0 (EIGEN_MPL2_ONLY enforces it)
#   nlohmann/json 3.12 MIT
#   CLI11 2.7.2        BSD-3-Clause
#   spdlog 1.17.0      MIT (bundles fmt, MIT)
#   doctest 2.5.3      MIT (tests only)
#   CUDA runtime       NVIDIA EULA, redistributable (found from the installed toolkit)

include(FetchContent)
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
set(FETCHCONTENT_QUIET ON)

# Third-party code is compiled with its own settings and treated as system headers, so
# GSO's warnings-as-errors apply only to GSO's own code.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "GSO links its dependencies statically" FORCE)

# --- Eigen: dense linear algebra and sparse containers --------------------------------
# Eigen's own CMake project builds tests and documentation targets; GSO only needs the
# headers, so the archive is fetched without running it and wrapped in a target here.
FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
    URL_HASH SHA256=e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec
    SOURCE_SUBDIR do-not-configure
    SYSTEM)
FetchContent_MakeAvailable(eigen)
add_library(gso_eigen INTERFACE)
add_library(Eigen3::Eigen ALIAS gso_eigen)
target_include_directories(gso_eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")
target_compile_definitions(gso_eigen INTERFACE EIGEN_MPL2_ONLY)

# --- nlohmann/json: study files and reports -------------------------------------------
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
    URL_HASH SHA256=4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187
    SYSTEM)

# --- CLI11: command-line parsing for apps/gso -----------------------------------------
set(CLI11_BUILD_TESTS OFF CACHE INTERNAL "")
set(CLI11_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(CLI11_BUILD_DOCS OFF CACHE INTERNAL "")
set(CLI11_INSTALL OFF CACHE INTERNAL "")
FetchContent_Declare(cli11
    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.7.2.tar.gz
    URL_HASH SHA256=46eef3101da70852ec7af026e09d485ccee81813331c8c6052d39344443b83da
    SYSTEM)

# --- spdlog: console and file log sinks in apps/gso -----------------------------------
set(SPDLOG_BUILD_EXAMPLE OFF CACHE INTERNAL "")
set(SPDLOG_BUILD_TESTS OFF CACHE INTERNAL "")
set(SPDLOG_INSTALL OFF CACHE INTERNAL "")
FetchContent_Declare(spdlog
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=d8862955c6d74e5846b3f580b1605d2428b11d97a410d86e2fb13e857cd3a744
    SYSTEM)

FetchContent_MakeAvailable(nlohmann_json cli11 spdlog)

# --- doctest: the few verification tests ----------------------------------------------
if(GSO_BUILD_TESTS)
    set(DOCTEST_WITH_TESTS OFF CACHE INTERNAL "")
    set(DOCTEST_NO_INSTALL ON CACHE INTERNAL "")
    FetchContent_Declare(doctest
        URL https://github.com/doctest/doctest/archive/refs/tags/v2.5.3.tar.gz
        URL_HASH SHA256=174ebc4e769928959614789c5b4e9c3d0a0f81a62bb608756b127bfebfb21331
        SYSTEM)
    FetchContent_MakeAvailable(doctest)
endif()

# --- CUDA runtime: GPU kernels ----------------------------------------------------------
find_package(CUDAToolkit 13.0 REQUIRED)
