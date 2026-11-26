// Volume and area of a hand-written 2 x 3 x 4 mm box OBJ (tests/data/box-2x3x4.obj):
// exactly 24 mm^3 and 2(6 + 8 + 12) = 52 mm^2. The file uses quads split into triangles,
// negative indices and two groups, so it also exercises the OBJ reader.

#include "gso/geometry/obj_file.hpp"

#include <doctest/doctest.h>

#include <filesystem>

TEST_CASE("A hand-written box has its exact volume and surface area")
{
    const auto bodies = gso::geometry::readObj(std::filesystem::path(GSO_TEST_DATA_DIR) / "box-2x3x4.obj");
    REQUIRE(bodies.size() == 2);  // the box is written as two groups: bottom-and-sides and top

    gso::geometry::TriangleMesh box;
    for (const auto& body : bodies) {
        const auto offset = static_cast<std::uint32_t>(box.vertices.size());
        box.vertices.insert(box.vertices.end(), body.mesh.vertices.begin(), body.mesh.vertices.end());
        for (auto triangle : body.mesh.triangles) {
            for (auto& index : triangle) {
                index += offset;
            }
            box.triangles.push_back(triangle);
        }
    }
    CHECK(gso::geometry::signedVolume(box) == doctest::Approx(24.0).epsilon(1e-14));
    CHECK(gso::geometry::surfaceArea(box) == doctest::Approx(52.0).epsilon(1e-14));
}
