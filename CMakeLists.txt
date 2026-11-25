cmake_minimum_required(VERSION 3.24)
project(GenerativeStructuralOptimizer VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
include(cmake/Dependencies.cmake)
add_library(gso_core STATIC
    src/geometry/implicit_solid.cpp
    src/geometry/quadrature.cpp
    src/mechanics/approximation.cpp
    src/mechanics/assembly.cpp
    src/mechanics/boundary.cpp
    src/mechanics/equilibrium.cpp
    src/optimization/refinement.cpp
    src/optimization/evolution.cpp)
target_include_directories(gso_core PUBLIC include)
target_include_directories(gso_core SYSTEM PUBLIC "${eigen_SOURCE_DIR}")
target_compile_definitions(gso_core PUBLIC EIGEN_MPL2_ONLY NOMINMAX)
add_executable(gso src/app/main.cpp src/io/read_study.cpp src/io/evolution_study.cpp)
target_link_libraries(gso PRIVATE gso_core)
target_include_directories(gso PRIVATE src/io)
target_include_directories(gso SYSTEM PRIVATE "${json_include}")
function(gso_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /external:W0 /utf-8 /bigobj)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endfunction()
gso_warnings(gso_core)
gso_warnings(gso)
include(CTest)
if(BUILD_TESTING)
    foreach(name IN ITEMS geometry approximation equilibrium refinement rod_convergence evolution)
        add_executable(test_${name} tests/${name}.cpp)
        target_link_libraries(test_${name} PRIVATE gso_core)
        gso_warnings(test_${name})
        add_test(NAME ${name} COMMAND test_${name})
    endforeach()
    add_test(NAME example_equilibrium COMMAND gso analyze "${CMAKE_CURRENT_SOURCE_DIR}/examples/elastic-block.json")
    add_test(NAME evolution_through_cli COMMAND "${CMAKE_COMMAND}" -DGSO=$<TARGET_FILE:gso>
        "-DSTUDY=${CMAKE_CURRENT_SOURCE_DIR}/examples/evolution-block.json"
        "-DOUT=${CMAKE_CURRENT_BINARY_DIR}/evolution-block-result.json" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/evolution_example.cmake")
    add_test(NAME invalid_study COMMAND gso analyze "${CMAKE_CURRENT_SOURCE_DIR}/tests/invalid-study.json")
    set_tests_properties(invalid_study PROPERTIES WILL_FAIL TRUE)
endif()
install(TARGETS gso RUNTIME DESTINATION bin)
