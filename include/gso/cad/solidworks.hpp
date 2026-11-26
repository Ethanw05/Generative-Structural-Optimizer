#pragma once

// Import of native SOLIDWORKS documents through the SOLIDWORKS COM API. This needs
// Windows with SOLIDWORKS installed; elsewhere the call throws EnvironmentError.

#include "gso/geometry/triangle_mesh.hpp"

#include <filesystem>
#include <vector>

namespace gso::cad {

// Tessellates a part (.SLDPRT) into one body, or an assembly (.SLDASM) into one body per
// visible component placed in assembly coordinates. Coordinates are millimetres.
//
// If the document is already open in SOLIDWORKS, that session's state is used and the
// document is left open; otherwise it is opened read-only and closed afterwards.
std::vector<geometry::NamedMesh> importSolidWorks(const std::filesystem::path& document);

}  // namespace gso::cad
