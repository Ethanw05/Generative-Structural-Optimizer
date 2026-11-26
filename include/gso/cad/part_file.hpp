#pragma once

// The single entry point for loading a part's geometry, whatever its file type:
// OBJ directly, SLDPRT and SLDASM through SOLIDWORKS.

#include "gso/geometry/triangle_mesh.hpp"

#include <filesystem>
#include <vector>

namespace gso::cad {

std::vector<geometry::NamedMesh> readPart(const std::filesystem::path& path);

}  // namespace gso::cad
