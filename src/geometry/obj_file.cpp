#include "gso/geometry/obj_file.hpp"

#include "gso/core/error.hpp"

#include <charconv>
#include <format>
#include <fstream>
#include <string>
#include <string_view>

namespace gso::geometry {
namespace {

// Reading state for one file. OBJ indices are global across groups (1-based, or negative
// relative to the most recent vertex), so vertices are collected once and each group's
// mesh receives a compact copy of the vertices it uses.
struct ObjReader {
    std::filesystem::path path;
    std::size_t lineNumber = 0;
    std::vector<Vec3> allVertices;
    struct Group {
        std::string name;
        std::vector<std::array<std::size_t, 3>> triangles;  // indices into allVertices
    };
    std::vector<Group> groups;

    [[noreturn]] void fail(const std::string& message) const
    {
        throw InputError(std::format("{}:{}", path.string(), lineNumber), message);
    }

    static std::string_view nextToken(std::string_view& rest)
    {
        const std::size_t start = rest.find_first_not_of(" \t\r");
        if (start == std::string_view::npos) {
            rest = {};
            return {};
        }
        const std::size_t end = rest.find_first_of(" \t\r", start);
        const std::string_view token = rest.substr(start, end - start);
        rest = (end == std::string_view::npos) ? std::string_view{} : rest.substr(end);
        return token;
    }

    double parseCoordinate(std::string_view token) const
    {
        double value = 0.0;
        const auto [end, status] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (status != std::errc{} || end != token.data() + token.size()) {
            fail(std::format("'{}' is not a number", token));
        }
        return value;
    }

    // Resolves the vertex part of a face corner ("12", "12/5", "12//3", "-1/2/3").
    std::size_t parseCorner(std::string_view token) const
    {
        const std::string_view vertexPart = token.substr(0, token.find('/'));
        long long index = 0;
        const auto [end, status] =
            std::from_chars(vertexPart.data(), vertexPart.data() + vertexPart.size(), index);
        if (status != std::errc{} || end != vertexPart.data() + vertexPart.size() || index == 0) {
            fail(std::format("'{}' is not a valid vertex reference", token));
        }
        const long long count = static_cast<long long>(allVertices.size());
        const long long resolved = index > 0 ? index - 1 : count + index;
        if (resolved < 0 || resolved >= count) {
            fail(std::format("vertex reference {} is outside the {} vertices read so far", index, count));
        }
        return static_cast<std::size_t>(resolved);
    }

    void readVertex(std::string_view rest)
    {
        Vec3 point;
        for (int axis = 0; axis < 3; ++axis) {
            const std::string_view token = nextToken(rest);
            if (token.empty()) {
                fail("a vertex needs three coordinates");
            }
            point[axis] = parseCoordinate(token);
        }
        allVertices.push_back(point);
    }

    void readFace(std::string_view rest)
    {
        std::array<std::size_t, 3> corners{};
        std::size_t count = 0;
        for (std::string_view token = nextToken(rest); !token.empty(); token = nextToken(rest)) {
            if (count == 3) {
                fail("only triangular faces are accepted; export the surface as triangles");
            }
            corners[count++] = parseCorner(token);
        }
        if (count != 3) {
            fail("a face needs three vertices");
        }
        if (groups.empty()) {
            groups.push_back({"body", {}});
        }
        groups.back().triangles.push_back(corners);
    }

    void readLine(std::string_view line)
    {
        std::string_view rest = line;
        const std::string_view keyword = nextToken(rest);
        if (keyword == "v") {
            readVertex(rest);
        } else if (keyword == "f") {
            readFace(rest);
        } else if (keyword == "o" || keyword == "g") {
            const std::size_t start = rest.find_first_not_of(" \t");
            std::string name(start == std::string_view::npos ? std::string_view{} : rest.substr(start));
            while (!name.empty() && (name.back() == '\r' || name.back() == ' ')) {
                name.pop_back();
            }
            groups.push_back({name.empty() ? "body" : name, {}});
        }
        // Comments, normals, texture coordinates, materials and smoothing groups are ignored.
    }

    std::vector<NamedMesh> finish() const
    {
        std::vector<NamedMesh> bodies;
        std::vector<std::uint32_t> compactIndex(allVertices.size());
        std::vector<std::size_t> stamp(allVertices.size(), 0);
        std::size_t groupNumber = 0;
        for (const Group& group : groups) {
            ++groupNumber;
            if (group.triangles.empty()) {
                continue;
            }
            NamedMesh body{group.name, {}};
            for (const auto& corners : group.triangles) {
                Triangle triangle{};
                for (std::size_t c = 0; c < 3; ++c) {
                    const std::size_t global = corners[c];
                    if (stamp[global] != groupNumber) {
                        stamp[global] = groupNumber;
                        compactIndex[global] = static_cast<std::uint32_t>(body.mesh.vertices.size());
                        body.mesh.vertices.push_back(allVertices[global]);
                    }
                    triangle[c] = compactIndex[global];
                }
                body.mesh.triangles.push_back(triangle);
            }
            bodies.push_back(std::move(body));
        }
        return bodies;
    }
};

}  // namespace

std::vector<NamedMesh> readObj(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw InputError(path.string(), "cannot open the file");
    }
    ObjReader reader{path};
    std::string line;
    while (std::getline(file, line)) {
        ++reader.lineNumber;
        reader.readLine(line);
    }
    std::vector<NamedMesh> bodies = reader.finish();
    if (bodies.empty()) {
        throw InputError(path.string(), "the file contains no triangles");
    }
    for (const NamedMesh& body : bodies) {
        validateIndices(body.mesh, std::format("{} ({})", path.string(), body.name));
    }
    return bodies;
}

void writeObj(const std::filesystem::path& path, const std::vector<NamedMesh>& bodies)
{
    namespace fs = std::filesystem;
    const fs::path temporary = fs::path(path).concat(".writing");
    try {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) {
            throw EnvironmentError(std::format("Cannot write {}", temporary.string()));
        }
        file << "# Generated by GSO. Coordinates in millimetres.\n";
        std::size_t vertexOffset = 1;  // OBJ indices are 1-based and global across groups.
        for (const NamedMesh& body : bodies) {
            file << "o " << body.name << '\n';
            for (const Vec3& v : body.mesh.vertices) {
                file << std::format("v {:.17g} {:.17g} {:.17g}\n", v.x(), v.y(), v.z());
            }
            for (const Triangle& t : body.mesh.triangles) {
                file << std::format("f {} {} {}\n", t[0] + vertexOffset, t[1] + vertexOffset,
                                    t[2] + vertexOffset);
            }
            vertexOffset += body.mesh.vertices.size();
        }
        file.close();
        if (!file) {
            throw EnvironmentError(std::format("Writing {} failed", temporary.string()));
        }
        fs::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
}

}  // namespace gso::geometry
