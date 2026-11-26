cmake_minimum_required(VERSION 3.24)
project(GenerativeStructuralOptimizer VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
include(cmake/Dependencies.cmake)

function(gso_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /external:W0 /utf-8 /bigobj)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endfunction()

# Numerical kernels shared by the executable and the libraries loaded from Python.
add_library(gso_core STATIC
    src/geometry/implicit_solid.cpp
    src/geometry/quadrature.cpp
    src/mechanics/approximation.cpp
    src/mechanics/assembly.cpp
    src/mechanics/transient.cpp
    src/mechanics/wrench.cpp
    src/mechanics/finite_strain.cpp
    src/optimization/surface_gradient.cpp
    src/optimization/feedback_sweep.cpp
    src/optimization/scenario_order.cpp)
target_include_directories(gso_core PUBLIC include)
target_include_directories(gso_core SYSTEM PUBLIC "${eigen_SOURCE_DIR}")
target_compile_definitions(gso_core PUBLIC EIGEN_MPL2_ONLY NOMINMAX)
set_target_properties(gso_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
gso_warnings(gso_core)

# Command-line helpers: CAD import, cell stiffness, surface motion and load-case order.
add_executable(gso src/app/main.cpp src/app/scenario_order.cpp src/io/surface_gradient.cpp src/io/cad/stencil.cpp)
target_link_libraries(gso PRIVATE gso_core)
target_include_directories(gso PRIVATE src/io)
if(WIN32 AND MSVC)
    target_sources(gso PRIVATE src/io/cad/automation.cpp src/io/cad/native.cpp)
    target_link_libraries(gso PRIVATE ole32 oleaut32 optimized comsuppw debug comsuppwd)
    target_compile_definitions(gso PRIVATE GSO_NATIVE_CAD)
endif()
gso_warnings(gso)

# Libraries loaded from Python with ctypes. They must sit beside the executable.
add_library(gso_field SHARED src/geometry/front_transport.cpp)
gso_warnings(gso_field)
add_library(gso_mechanics SHARED src/io/finite_strain_api.cpp src/io/wrench_api.cpp src/io/history_api.cpp src/mechanics/history_envelope.cpp)
target_link_libraries(gso_mechanics PRIVATE gso_core)
gso_warnings(gso_mechanics)
add_library(gso_evolution SHARED src/io/feedback_api.cpp)
target_link_libraries(gso_evolution PRIVATE gso_core)
gso_warnings(gso_evolution)
add_dependencies(gso gso_field gso_mechanics gso_evolution)

include(CTest)
if(BUILD_TESTING)
    foreach(name IN ITEMS approximation finite_strain scenario_feedback)
        add_executable(test_${name} tests/${name}.cpp)
        target_link_libraries(test_${name} PRIVATE gso_core)
        gso_warnings(test_${name})
        add_test(NAME ${name} COMMAND test_${name})
    endforeach()
    # The Python tests need requirements.txt installed in the interpreter CMake finds.
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_workflow_dependencies.py"
            RESULT_VARIABLE gso_workflow_status
            ERROR_VARIABLE gso_workflow_error
            ERROR_STRIP_TRAILING_WHITESPACE)
        if(gso_workflow_status STREQUAL "0")
            foreach(workflow_test IN ITEMS shape_adjoint stress_evolution design_inputs)
                add_test(NAME ${workflow_test} COMMAND "${Python3_EXECUTABLE}" -B -X faulthandler
                         "${CMAKE_CURRENT_SOURCE_DIR}/tests/${workflow_test}.py" $<TARGET_FILE:gso>)
            endforeach()
        else()
            message(STATUS "Python tests skipped. Install requirements.txt to enable them. ${gso_workflow_error}")
        endif()
    endif()
endif()
install(TARGETS gso gso_field gso_mechanics gso_evolution RUNTIME DESTINATION bin LIBRARY DESTINATION bin)
