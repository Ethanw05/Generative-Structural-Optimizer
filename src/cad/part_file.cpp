#include "gso/cad/part_file.hpp"

#include "gso/cad/solidworks.hpp"
#include "gso/core/error.hpp"
#include "gso/geometry/obj_file.hpp"

#include <algorithm>
#include <cctype>

namespace gso::cad {

std::vector<geometry::NamedMesh> readPart(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension == ".obj") {
        return geometry::readObj(path);
    }
    if (extension == ".sldprt" || extension == ".sldasm") {
        return importSolidWorks(path);
    }
    throw InputError(path.string(), "unsupported part format; use OBJ, SLDPRT or SLDASM");
}

}  // namespace gso::cad
