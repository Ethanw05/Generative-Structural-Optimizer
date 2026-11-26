# GSO's modules, lowest layer first. Each may depend only on the modules above it in
# this file (docs/port-plan.md, section 5).

configure_file(core/version.hpp.in "${PROJECT_BINARY_DIR}/generated/gso/core/version.hpp" @ONLY)

gso_module(core
    SOURCES
        core/error.cpp
        core/log.cpp
        core/progress.cpp
        core/sha256.cpp)
target_include_directories(gso_core PUBLIC "${PROJECT_BINARY_DIR}/generated")

gso_module(geometry
    SOURCES
        geometry/triangle_mesh.cpp
        geometry/obj_file.cpp
    PUBLIC gso::core Eigen3::Eigen)

gso_module(solvers
    SOURCES
        solvers/gpu_device.cu
    PUBLIC gso::core
    PRIVATE CUDA::cudart_static)

gso_module(study
    SOURCES
        study/json_fields.cpp
        study/material_reader.cpp
        study/study_file.cpp
    PUBLIC gso::core Eigen3::Eigen
    PRIVATE nlohmann_json::nlohmann_json)

if(WIN32)
    set(gso_solidworks_sources cad/com_automation.cpp cad/solidworks_windows.cpp)
    set(gso_solidworks_libraries ole32 oleaut32 $<IF:$<CONFIG:Debug>,comsuppwd,comsuppw>)
else()
    set(gso_solidworks_sources cad/solidworks_unavailable.cpp)
    set(gso_solidworks_libraries)
endif()
gso_module(cad
    SOURCES
        cad/part_file.cpp
        ${gso_solidworks_sources}
    PUBLIC gso::geometry
    PRIVATE ${gso_solidworks_libraries})
