#include "gso/geometry/triangle_mesh.hpp"

#include "gso/core/error.hpp"

#include <Eigen/Geometry>

#include <cmath>
#include <format>
#include <limits>
#include <span>

namespace gso::geometry {

void validateIndices(const TriangleMesh& mesh, const std::string& where)
{
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        if (!mesh.vertices[i].allFinite()) {
            throw InputError(where, std::format("vertex {} has a non-finite coordinate", i));
        }
    }
    const std::size_t vertexCount = mesh.vertices.size();
    for (std::size_t t = 0; t < mesh.triangles.size(); ++t) {
        for (const std::uint32_t index : mesh.triangles[t]) {
            if (index >= vertexCount) {
                throw InputError(where, std::format("triangle {} refers to vertex {}, but the mesh has {}",
                                                    t, index, vertexCount));
            }
        }
    }
}

BoundingBox boundingBox(const TriangleMesh& mesh)
{
    constexpr double kInfinity = std::numeric_limits<double>::infinity();
    BoundingBox box{Vec3::Constant(kInfinity), Vec3::Constant(-kInfinity)};
    for (const Vec3& vertex : mesh.vertices) {
        box.min = box.min.cwiseMin(vertex);
        box.max = box.max.cwiseMax(vertex);
    }
    return box;
}

double signedVolume(const TriangleMesh& mesh)
{
    // Sum of signed tetrahedra formed by each triangle and the origin: V = sum(a . (b x c)) / 6.
    double sixTimesVolume = 0.0;
    for (const Triangle& triangle : mesh.triangles) {
        const Vec3& a = mesh.vertices[triangle[0]];
        const Vec3& b = mesh.vertices[triangle[1]];
        const Vec3& c = mesh.vertices[triangle[2]];
        sixTimesVolume += a.dot(b.cross(c));
    }
    return sixTimesVolume / 6.0;
}

double surfaceArea(const TriangleMesh& mesh)
{
    double twiceArea = 0.0;
    for (const Triangle& triangle : mesh.triangles) {
        const Vec3& a = mesh.vertices[triangle[0]];
        const Vec3 ab = mesh.vertices[triangle[1]] - a;
        const Vec3 ac = mesh.vertices[triangle[2]] - a;
        twiceArea += ab.cross(ac).norm();
    }
    return twiceArea / 2.0;
}

Digest contentDigest(const TriangleMesh& mesh)
{
    Sha256 hasher;
    // The counts are hashed first so that a different split between vertices and
    // triangles cannot produce the same byte stream.
    const std::array<std::uint64_t, 2> counts = {mesh.vertices.size(), mesh.triangles.size()};
    hasher.update(std::as_bytes(std::span(counts)));
    for (const Vec3& vertex : mesh.vertices) {
        const std::array<double, 3> xyz = {vertex.x(), vertex.y(), vertex.z()};
        hasher.update(std::as_bytes(std::span(xyz)));
    }
    hasher.update(std::as_bytes(std::span(mesh.triangles)));
    return hasher.finish();
}

}  // namespace gso::geometry
