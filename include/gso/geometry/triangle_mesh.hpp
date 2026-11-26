#pragma once

// Closed triangle surfaces: imported CAD parts, generated designs and design-space solids.
// Units are millimetres throughout GSO.

#include "gso/core/sha256.hpp"

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace gso::geometry {

using Vec3 = Eigen::Vector3d;
using Triangle = std::array<std::uint32_t, 3>;

struct BoundingBox {
    Vec3 min;
    Vec3 max;
};

// An indexed triangle mesh. Triangles are wound counter-clockwise when seen from outside,
// so normals point out of the material.
struct TriangleMesh {
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;

    bool empty() const { return triangles.empty(); }
};

// A mesh with a name: one body of a part, or one component of an assembly.
struct NamedMesh {
    std::string name;
    TriangleMesh mesh;
};

// Throws InputError (naming `where`) if any index is out of range or any coordinate is
// not finite. Every mesh entering GSO from outside passes through this check.
void validateIndices(const TriangleMesh& mesh, const std::string& where);

BoundingBox boundingBox(const TriangleMesh& mesh);

// Enclosed volume by the divergence theorem. Positive for a closed, outward-wound mesh.
double signedVolume(const TriangleMesh& mesh);

double surfaceArea(const TriangleMesh& mesh);

// Content digest of the exact vertex coordinates and triangle indices. Two meshes have
// the same digest only if they are bit-for-bit identical.
Digest contentDigest(const TriangleMesh& mesh);

}  // namespace gso::geometry
