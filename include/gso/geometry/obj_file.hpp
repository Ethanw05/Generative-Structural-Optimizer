#pragma once

// Wavefront OBJ reading and writing. OBJ is GSO's exchange format for triangle surfaces:
// SOLIDWORKS imports, generated parts and design-space solids. Each `o` or `g` group
// becomes one NamedMesh. Only geometry is read (v and f records); texture coordinates,
// normals and materials are ignored. Coordinates are millimetres.

#include "gso/geometry/triangle_mesh.hpp"

#include <filesystem>
#include <vector>

namespace gso::geometry {

// Reads every group in the file. Polygons with more than three corners are rejected
// rather than triangulated, because fan triangulation is wrong for non-convex faces.
// Throws InputError naming the file and line of any malformed record.
std::vector<NamedMesh> readObj(const std::filesystem::path& path);

// Writes the bodies with round-trip precision (17 significant digits). The file is
// written to a temporary name and renamed into place, so a reader never sees a partial
// file and a failed write leaves any previous file untouched.
void writeObj(const std::filesystem::path& path, const std::vector<NamedMesh>& bodies);

}  // namespace gso::geometry
