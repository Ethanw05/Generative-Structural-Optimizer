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
    src/mechanics/transient.cpp
    src/mechanics/stress.cpp
    src/mechanics/stress_adjacency.cpp
    src/mechanics/stress_regions.cpp
    src/mechanics/wrench.cpp
    src/mechanics/finite_strain.cpp
    src/mechanics/equilibrium.cpp
    src/optimization/surface_gradient.cpp
    src/optimization/boundary_step.cpp
    src/optimization/refinement.cpp
    src/optimization/feedback_sweep.cpp
    src/optimization/scenario_order.cpp
    src/optimization/global_starts.cpp
    src/optimization/evolution.cpp)
target_include_directories(gso_core PUBLIC include)
target_include_directories(gso_core SYSTEM PUBLIC "${eigen_SOURCE_DIR}")
target_compile_definitions(gso_core PUBLIC EIGEN_MPL2_ONLY NOMINMAX)
add_executable(gso src/app/main.cpp src/app/scenario_order.cpp src/io/boundary_step.cpp src/io/surface_gradient.cpp src/io/cad/stencil.cpp src/io/read_study.cpp src/io/scenario.cpp src/io/evolution_study.cpp)
target_link_libraries(gso PRIVATE gso_core)
if(WIN32 AND MSVC)
    target_sources(gso PRIVATE src/io/cad/automation.cpp src/io/cad/native.cpp)
    target_link_libraries(gso PRIVATE ole32 oleaut32 optimized comsuppw debug comsuppwd)
    target_compile_definitions(gso PRIVATE GSO_NATIVE_CAD)
endif()
target_include_directories(gso PRIVATE src/io)
target_include_directories(gso SYSTEM PRIVATE "${json_include}")
function(gso_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /external:W0 /utf-8 /bigobj)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endfunction()
add_library(gso_field SHARED src/geometry/front_transport.cpp)
add_library(gso_view SHARED src/visualization/orthographic.cpp)
gso_warnings(gso_view)
add_library(gso_mechanics SHARED src/io/finite_strain_api.cpp src/io/wrench_api.cpp src/io/history_api.cpp src/mechanics/history_envelope.cpp)
target_link_libraries(gso_mechanics PRIVATE gso_core)
gso_warnings(gso_mechanics)
add_library(gso_evolution SHARED src/io/feedback_api.cpp)
target_link_libraries(gso_evolution PRIVATE gso_core)
gso_warnings(gso_evolution)
add_dependencies(gso gso_field gso_view gso_mechanics gso_evolution)
gso_warnings(gso_field)
gso_warnings(gso_core)
gso_warnings(gso)
include(CTest)
if(BUILD_TESTING)
    foreach(name IN ITEMS geometry approximation equilibrium refinement rod_convergence evolution transient boundary_step stress_watershed scenario_feedback finite_strain)
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
    option(GSO_REQUIRE_WORKFLOW_TESTS "Fail configuration if the Python workflow dependencies are unavailable" OFF)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    set(gso_workflow_status 1)
    set(gso_workflow_error "Python interpreter unavailable")
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_workflow_dependencies.py"
            RESULT_VARIABLE gso_workflow_status
            ERROR_VARIABLE gso_workflow_error
            ERROR_STRIP_TRAILING_WHITESPACE)
    endif()
    if(gso_workflow_status STREQUAL "0")
        foreach(workflow_test IN ITEMS design_targets graded_load_transfer seeded_structures front_curvature design_inputs stress_evolution shape_adjoint stress_regions perforated_plate regional_history cad_history nonlinear_cad analysis_mesh orthographic realized_shape)
            add_test(NAME ${workflow_test} COMMAND "${Python3_EXECUTABLE}" -B -X faulthandler
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/${workflow_test}.py" $<TARGET_FILE:gso>)
        endforeach()
        # This end-to-end study independently remeshes two generations under two loads.
        set_tests_properties(realized_shape PROPERTIES TIMEOUT 7200)
    elseif(GSO_REQUIRE_WORKFLOW_TESTS)
        message(FATAL_ERROR "Workflow dependencies unavailable. Install requirements.txt into Python3_EXECUTABLE.\n${gso_workflow_error}")
    else()
        message(STATUS "Building native tests only. Install requirements.txt to enable workflow tests. ${gso_workflow_error}")
    endif()
endif()
install(TARGETS gso gso_field gso_view gso_mechanics gso_evolution RUNTIME DESTINATION bin LIBRARY DESTINATION bin)
