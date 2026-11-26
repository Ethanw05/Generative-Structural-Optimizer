cmake_minimum_required(VERSION 3.28)
project(GenerativeStructuralOptimizer
    VERSION 0.2.0
    DESCRIPTION "Generative structural optimizer"
    LANGUAGES C CXX CUDA)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "GSO requires a 64-bit build.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# The dynamic C++ runtime is required by oneTBB. Every target uses the same runtime.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

option(GSO_BUILD_TESTS "Build the verification tests" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GsoDependencies)
include(GsoModule)

add_subdirectory(src)
add_subdirectory(apps/gso)

if(GSO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
