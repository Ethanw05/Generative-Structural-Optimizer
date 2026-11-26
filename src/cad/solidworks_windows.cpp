// SOLIDWORKS import on Windows, through the SOLIDWORKS COM API.

#include "gso/cad/solidworks.hpp"

#include "cad/com_automation.hpp"
#include "gso/core/error.hpp"

#include <algorithm>
#include <array>
#include <cwctype>
#include <format>
#include <map>

namespace gso::cad {
namespace {

using geometry::NamedMesh;
using geometry::TriangleMesh;
using geometry::Vec3;

// COM interface identifiers from the SOLIDWORKS type library.
constexpr const wchar_t* kPartDocInterface = L"{83A33D32-27C5-11CE-BFD4-00400513BB57}";
constexpr const wchar_t* kAssemblyDocInterface = L"{83A33D35-27C5-11CE-BFD4-00400513BB57}";

// swDocumentTypes_e values.
constexpr long kPartDocument = 1;
constexpr long kAssemblyDocument = 2;
// swOpenDocOptions_e: Silent (1) | ReadOnly (2).
constexpr long kOpenSilentReadOnly = 3;

// SOLIDWORKS works in metres; GSO works in millimetres.
constexpr double kMillimetresPerMetre = 1000.0;

// A component placement from IMathTransform::ArrayData: a 3x3 rotation stored by rows
// in elements 0-8, a translation in 9-11 and a uniform scale in 12. A point p in the
// component maps to   translation + scale * (p.x * row0 + p.y * row1 + p.z * row2).
struct Placement {
    std::array<double, 13> data{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};

    Vec3 apply(const Vec3& p) const
    {
        Vec3 result;
        for (int axis = 0; axis < 3; ++axis) {
            const double rotated = p.x() * data[axis] + p.y() * data[3 + axis] + p.z() * data[6 + axis];
            result[axis] = data[9 + axis] + data[12] * rotated;
        }
        return result;
    }

    // A mirroring placement reverses triangle winding, which must be undone so normals
    // keep pointing out of the material.
    bool mirrors() const
    {
        const auto& t = data;
        const double determinant = t[12] * (t[0] * (t[4] * t[8] - t[5] * t[7]) -
                                            t[1] * (t[3] * t[8] - t[5] * t[6]) +
                                            t[2] * (t[3] * t[7] - t[4] * t[6]));
        return determinant < 0.0;
    }
};

// Releases a document GSO opened itself. A document that was already open in the user's
// SOLIDWORKS session is left alone.
class OpenedDocument {
public:
    OpenedDocument(com::Dispatch application, com::Dispatch model, bool ownedByGso)
        : application_(std::move(application)), model_(std::move(model)), owned_(ownedByGso)
    {
        if (owned_) {
            title_ = com::call(model_, L"GetTitle");
        }
    }
    ~OpenedDocument()
    {
        if (owned_) {
            try {
                com::call(application_, L"CloseDoc", {title_});
            } catch (...) {
                // Closing is best effort; the import result is already complete.
            }
        }
    }
    OpenedDocument(const OpenedDocument&) = delete;
    OpenedDocument& operator=(const OpenedDocument&) = delete;

    const com::Dispatch& model() const { return model_; }
    bool ownedByGso() const { return owned_; }

private:
    com::Dispatch application_;
    com::Dispatch model_;
    _variant_t title_;
    bool owned_;
};

long documentKind(const std::filesystem::path& path)
{
    std::wstring extension = path.extension().wstring();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
    if (extension == L".sldprt") {
        return kPartDocument;
    }
    if (extension == L".sldasm") {
        return kAssemblyDocument;
    }
    throw InputError(path.string(), "expected a SOLIDWORKS part (.SLDPRT) or assembly (.SLDASM)");
}

com::Dispatch startSolidWorks()
{
    CLSID classId{};
    com::check(CLSIDFromProgID(L"SldWorks.Application", &classId), "Find the installed SOLIDWORKS");
    com::Dispatch application;
    com::check(CoCreateInstance(classId, nullptr, CLSCTX_LOCAL_SERVER, IID_PPV_ARGS(&application)),
               "Start SOLIDWORKS");
    return application;
}

OpenedDocument openDocument(const com::Dispatch& application, const std::filesystem::path& path, long kind)
{
    const _variant_t name(path.c_str());
    com::Dispatch model = com::asObject(com::call(application, L"GetOpenDocumentByName", {name}));
    if (model) {
        return OpenedDocument(application, model, false);
    }
    long errors = 0;
    long warnings = 0;
    _variant_t errorsOut;
    _variant_t warningsOut;
    errorsOut.vt = VT_BYREF | VT_I4;
    errorsOut.plVal = &errors;
    warningsOut.vt = VT_BYREF | VT_I4;
    warningsOut.plVal = &warnings;
    model = com::asObject(com::call(application, L"OpenDoc6",
                                    {name, _variant_t(kind), _variant_t(kOpenSilentReadOnly), _variant_t(L""),
                                     errorsOut, warningsOut}));
    if (!model) {
        throw EnvironmentError(std::format("SOLIDWORKS could not open {} (error code {})", path.string(), errors));
    }
    return OpenedDocument(application, model, true);
}

// Builds an indexed mesh from SOLIDWORKS tessellation: a flat array of 9 coordinates
// (three corners) per triangle, in metres. Corners with identical coordinates are welded
// so the result is a connected surface.
TriangleMesh meshFromTessellation(const std::vector<double>& corners, const Placement& placement,
                                  const std::string& bodyName)
{
    if (corners.empty() || corners.size() % 9 != 0) {
        throw EnvironmentError(std::format("SOLIDWORKS returned an incomplete surface for {}", bodyName));
    }
    TriangleMesh mesh;
    std::map<std::array<double, 3>, std::uint32_t> vertexIds;
    std::vector<std::uint32_t> cornerIds;
    cornerIds.reserve(corners.size() / 3);
    for (std::size_t i = 0; i < corners.size(); i += 3) {
        const Vec3 local(corners[i], corners[i + 1], corners[i + 2]);
        const Vec3 point = kMillimetresPerMetre * placement.apply(local);
        if (!point.allFinite()) {
            throw EnvironmentError(std::format("SOLIDWORKS returned a non-finite point for {}", bodyName));
        }
        const auto [entry, inserted] =
            vertexIds.emplace(std::array{point.x(), point.y(), point.z()}, static_cast<std::uint32_t>(mesh.vertices.size()));
        if (inserted) {
            mesh.vertices.push_back(point);
        }
        cornerIds.push_back(entry->second);
    }
    const bool flip = placement.mirrors();
    for (std::size_t i = 0; i < cornerIds.size(); i += 3) {
        mesh.triangles.push_back(flip ? geometry::Triangle{cornerIds[i], cornerIds[i + 2], cornerIds[i + 1]}
                                      : geometry::Triangle{cornerIds[i], cornerIds[i + 1], cornerIds[i + 2]});
    }
    return mesh;
}

std::vector<NamedMesh> importPart(const OpenedDocument& document, const std::filesystem::path& path)
{
    const com::Dispatch part = com::asInterface(document.model(), kPartDocInterface);
    const std::vector<double> corners = com::asNumberArray(com::call(part, L"GetTessTriangles", {_variant_t(true)}));
    const std::string name = path.stem().string();
    return {NamedMesh{name, meshFromTessellation(corners, Placement{}, name)}};
}

// Some components report no tessellation of their own (for example when the assembly
// cuts them). Their resolved, in-context bodies are then tessellated face by face.
std::vector<double> componentTessellation(const com::Dispatch& component)
{
    const _variant_t direct = com::call(component, L"GetTessTriangles", {_variant_t(true)});
    if ((direct.vt & VT_ARRAY) != 0) {
        std::vector<double> corners = com::asNumberArray(direct);
        if (!corners.empty()) {
            return corners;
        }
    }
    std::vector<double> corners;
    _variant_t bodyInfo;
    _variant_t bodyInfoOut;
    bodyInfoOut.vt = VT_VARIANT | VT_BYREF;
    bodyInfoOut.pvarVal = &bodyInfo;
    constexpr long kSolidBodies = 0;
    for (const com::Dispatch& body : com::asObjectArray(com::call(component, L"GetBodies3", {_variant_t(kSolidBodies), bodyInfoOut}))) {
        for (const com::Dispatch& face : com::asObjectArray(com::call(body, L"GetFaces"))) {
            const _variant_t triangles = com::call(face, L"GetTessTriangles", {_variant_t(true)});
            if ((triangles.vt & VT_ARRAY) != 0) {
                const std::vector<double> faceCorners = com::asNumberArray(triangles);
                corners.insert(corners.end(), faceCorners.begin(), faceCorners.end());
            }
        }
    }
    return corners;
}

std::vector<NamedMesh> importAssembly(const OpenedDocument& document)
{
    const com::Dispatch assembly = com::asInterface(document.model(), kAssemblyDocInterface);
    if (document.ownedByGso()) {
        com::call(assembly, L"ResolveAllLightWeightComponents", {_variant_t(false)});
    }
    std::vector<NamedMesh> bodies;
    // GetComponents(ToplevelOnly = false) returns the components at every level of the tree.
    constexpr bool kTopLevelOnly = false;
    for (const com::Dispatch& component : com::asObjectArray(com::call(assembly, L"GetComponents", {_variant_t(kTopLevelOnly)}))) {
        if (static_cast<bool>(com::call(component, L"IsSuppressed"))) {
            continue;
        }
        const std::string name = std::filesystem::path(com::asString(com::call(component, L"Name2"))).string();
        const com::Dispatch child = com::asObject(com::call(component, L"GetModelDoc2"));
        if (!child) {
            throw EnvironmentError(std::format("Assembly component {} could not be resolved", name));
        }
        if (static_cast<long>(com::call(child, L"GetType")) == kAssemblyDocument) {
            continue;  // Sub-assemblies contribute through their own part components.
        }
        Placement placement;
        const std::vector<double> transform =
            com::asNumberArray(com::call(com::asObject(com::call(component, L"Transform2")), L"ArrayData"));
        if (transform.size() != 16) {
            throw EnvironmentError(std::format("Component {} has an invalid placement", name));
        }
        std::copy_n(transform.begin(), placement.data.size(), placement.data.begin());
        bodies.push_back({name, meshFromTessellation(componentTessellation(component), placement, name)});
    }
    return bodies;
}

}  // namespace

std::vector<NamedMesh> importSolidWorks(const std::filesystem::path& documentPath)
{
    if (!std::filesystem::exists(documentPath)) {
        throw InputError(documentPath.string(), "file not found");
    }
    const std::filesystem::path path = std::filesystem::canonical(documentPath);
    const long kind = documentKind(path);
    try {
        com::Apartment apartment;
        const com::Dispatch application = startSolidWorks();
        const OpenedDocument document = openDocument(application, path, kind);
        std::vector<NamedMesh> bodies = kind == kPartDocument ? importPart(document, path) : importAssembly(document);
        if (bodies.empty()) {
            throw InputError(path.string(), "the document has no solid geometry");
        }
        return bodies;
    } catch (const _com_error& error) {
        throw EnvironmentError(std::format("SOLIDWORKS automation failed (HRESULT 0x{:08x})",
                                           static_cast<unsigned long>(error.Error())));
    }
}

}  // namespace gso::cad
