# Verification tests. Each one checks GSO against something independent of GSO: a
# published reference value, an exact analytic answer or a closed-form bound. Tests that
# only restate the implementation are deliberately absent.

add_executable(gso_verification
    main.cpp
    sha256_vectors.cpp
    mesh_measures.cpp
    material_admissibility.cpp)
target_link_libraries(gso_verification PRIVATE gso::geometry gso::study doctest::doctest)
target_compile_definitions(gso_verification PRIVATE GSO_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
gso_apply_warnings(gso_verification)
add_test(NAME verification COMMAND gso_verification)

# Every study shipped in studies/ must stay valid as the format evolves.
foreach(study IN ITEMS coupon necked-link)
    add_test(NAME study-${study} COMMAND gso check "${PROJECT_SOURCE_DIR}/studies/${study}/study.json")
endforeach()
