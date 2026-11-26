// SOLIDWORKS import on platforms without SOLIDWORKS. The COM API exists only on Windows.

#include "gso/cad/solidworks.hpp"

#include "gso/core/error.hpp"

namespace gso::cad {

std::vector<geometry::NamedMesh> importSolidWorks(const std::filesystem::path& document)
{
    throw EnvironmentError("Importing " + document.string() +
                           " needs SOLIDWORKS, which runs only on Windows. Export the part as OBJ instead.");
}

}  // namespace gso::cad
