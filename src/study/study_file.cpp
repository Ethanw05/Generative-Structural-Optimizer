#include "gso/study/study_file.hpp"

#include "study/json_fields.hpp"
#include "study/material_reader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <fstream>
#include <set>
#include <sstream>
#include <utility>

namespace gso::study {

const std::string& caseName(const LoadCase& loadCase)
{
    return std::visit([](const auto& c) -> const std::string& { return c.name; }, loadCase);
}

namespace {

using detail::Fields;
using detail::Json;

// ---- Small parsing helpers ---------------------------------------------------------------

template <typename Enum, std::size_t N>
Enum parseName(Fields& fields, const std::string& key, const std::array<std::pair<const char*, Enum>, N>& names)
{
    const std::string value = fields.text(key);
    std::string allowed;
    for (const auto& [name, item] : names) {
        if (value == name) {
            return item;
        }
        allowed += allowed.empty() ? name : std::string(", ") + name;
    }
    fields.fail(key, std::format("unknown value '{}'; expected one of: {}", value, allowed));
}

Digest parseDigest(Fields& fields, const std::string& key)
{
    const std::string hex = fields.text(key);
    Digest digest{};
    if (hex.size() != 2 * digest.size()) {
        fields.fail(key, "expected 64 hexadecimal characters");
    }
    for (std::size_t i = 0; i < digest.size(); ++i) {
        unsigned value = 0;
        std::istringstream byte(hex.substr(2 * i, 2));
        if (!(byte >> std::hex >> value)) {
            fields.fail(key, "expected 64 hexadecimal characters");
        }
        digest[i] = static_cast<std::uint8_t>(value);
    }
    return digest;
}

std::filesystem::path resolve(const std::filesystem::path& base, const std::string& relative)
{
    return std::filesystem::absolute(base / std::filesystem::path(relative)).lexically_normal();
}

Vec3 unitVector(Fields& fields, const std::string& key)
{
    const Vec3 v = fields.vector3(key);
    if (v.norm() < 1e-12) {
        fields.fail(key, "must not be the zero vector");
    }
    return v.normalized();
}

// ---- Header, part and frame ------------------------------------------------------------------

void readHeader(Fields& root, Study& study)
{
    if (root.text("format") != "gso-study") {
        root.fail("format", "expected \"gso-study\"");
    }
    study.formatVersion = root.integer("version");
    if (study.formatVersion < 1 || study.formatVersion > kCurrentFormatVersion) {
        root.fail("version", std::format("this build reads study versions 1 to {}, found {}",
                                         kCurrentFormatVersion, study.formatVersion));
    }
    // The units are fixed. Stating them in the file makes every study self-describing.
    Fields units = root.object("units");
    const std::array<std::pair<const char*, const char*>, 5> expected = {
        {{"length", "mm"}, {"force", "N"}, {"time", "s"}, {"stress", "MPa"}, {"mass", "kg"}}};
    for (const auto& [quantity, unit] : expected) {
        if (units.text(quantity) != unit) {
            units.fail(quantity, std::format("GSO studies use {} for {}", unit, quantity));
        }
    }
    units.finish();
}

void readPart(Fields& root, Study& study, const std::filesystem::path& base)
{
    Fields part = root.object("part");
    study.part = resolve(base, part.text("source"));
    if (part.has("sha256")) {
        study.partDigest = parseDigest(part, "sha256");
    }
    part.finish();
}

LocalFrame readFrame(Fields fields)
{
    LocalFrame frame;
    frame.origin = fields.vector3("origin");
    frame.xAxis = unitVector(fields, "x_axis");
    frame.upAxis = unitVector(fields, "up_axis");
    // A frame with skewed axes would silently distort every load entered in it.
    if (std::abs(frame.xAxis.dot(frame.upAxis)) > 1e-6) {
        fields.fail("up_axis", "must be perpendicular to x_axis");
    }
    fields.finish();
    return frame;
}

// ---- Selections and design space ---------------------------------------------------------

void readSelections(Fields& root, Study& study)
{
    Fields selections = root.object("selections");
    const Json& entries = root.raw("selections");
    for (const auto& [name, value] : entries.items()) {
        Fields entry = selections.object(name);
        Fields face = entry.object("planar_face");
        Selection selection{name, {}};
        selection.planarFace.point = face.vector3("point");
        selection.planarFace.normal = unitVector(face, "normal");
        if (face.has("tolerance_mm")) {
            selection.planarFace.toleranceMm = face.positive("tolerance_mm");
        }
        face.finish();
        entry.finish();
        study.selections.emplace(name, selection);
    }
    selections.finish();
}

void requireSelection(const Study& study, Fields& fields, const std::string& key, const std::string& name)
{
    if (!study.selections.contains(name)) {
        fields.fail(key, std::format("no selection named '{}'", name));
    }
}

void readDesignSpace(Fields& root, Study& study, const std::filesystem::path& base)
{
    Fields space = root.object("design_space");
    if (space.has("envelope") && space.raw("envelope").is_object()) {
        auto [kind, box] = space.singleVariant("envelope");
        if (kind != "box") {
            box.fail("", std::format("unknown envelope '{}'; expected \"bounding_box\", \"part\" or {{\"box\": ...}}", kind));
        }
        study.designSpace.envelope = Envelope::Box;
        study.designSpace.boxMinMm = box.vector3("min_mm");
        study.designSpace.boxMaxMm = box.vector3("max_mm");
        if ((study.designSpace.boxMaxMm.array() <= study.designSpace.boxMinMm.array()).any()) {
            box.fail("max_mm", "must exceed min_mm along every axis");
        }
        box.finish();
    } else {
        study.designSpace.envelope =
            parseName(space, "envelope", std::array{std::pair{"bounding_box", Envelope::BoundingBox},
                                                    std::pair{"part", Envelope::Part}});
    }
    for (Fields region : space.objectList("preserve")) {
        PreservedRegion preserved{region.text("selection"), region.positive("depth_mm")};
        requireSelection(study, region, "selection", preserved.selection);
        region.finish();
        study.designSpace.preserved.push_back(preserved);
    }
    for (Fields exclusion : space.objectList("exclude")) {
        study.designSpace.excludedSolids.push_back(resolve(base, exclusion.text("solid")));
        exclusion.finish();
    }
    space.finish();
}

// ---- Materials and cases -----------------------------------------------------------------

void readMaterials(Fields& root, Study& study)
{
    Fields materials = root.object("materials");
    for (const auto& [name, value] : root.raw("materials").items()) {
        study.materials.emplace(name, detail::readMaterial(name, materials.object(name)));
    }
    if (study.materials.empty()) {
        root.fail("materials", "at least one material is required");
    }
    materials.finish();
}

LoadFrame readLoadFrame(Fields& load, const Study& study)
{
    if (!load.has("frame")) {
        return LoadFrame::Global;
    }
    const LoadFrame frame = parseName(load, "frame", std::array{std::pair{"global", LoadFrame::Global},
                                                                std::pair{"local", LoadFrame::Local}});
    if (frame == LoadFrame::Local && !study.frame) {
        load.fail("frame", "a local load needs the study's \"frame\" to be defined");
    }
    return frame;
}

Load readLoad(Fields load, const Study& study)
{
    const std::string on = load.text("on");
    requireSelection(study, load, "on", on);
    const LoadFrame frame = readLoadFrame(load, study);
    const bool isForce = load.has("force_n");
    const bool isTorque = load.has("torque_nmm");
    if (isForce == isTorque) {
        load.fail("", "a load needs exactly one of force_n or torque_nmm");
    }
    Load result = isForce ? Load{ForceLoad{on, load.vector3("force_n"), frame}}
                          : Load{TorqueLoad{on, load.vector3("torque_nmm"), frame}};
    load.finish();
    return result;
}

StaticCase readStaticCase(Fields& entry, const Study& study)
{
    StaticCase loadCase;
    loadCase.name = entry.text("name");
    loadCase.material = entry.text("material");
    if (!study.materials.contains(loadCase.material)) {
        entry.fail("material", std::format("no material named '{}'", loadCase.material));
    }
    loadCase.fixtures = entry.textList("fixtures");
    if (loadCase.fixtures.empty()) {
        entry.fail("fixtures", "a static case needs at least one fixture");
    }
    for (const std::string& fixture : loadCase.fixtures) {
        requireSelection(study, entry, "fixtures", fixture);
    }
    for (Fields load : entry.objectList("loads")) {
        loadCase.loads.push_back(readLoad(load, study));
    }
    if (loadCase.loads.empty()) {
        entry.fail("loads", "a static case needs at least one load");
    }
    return loadCase;
}

void readCases(Fields& root, Study& study)
{
    std::set<std::string> names;
    for (Fields entry : root.objectList("cases")) {
        const std::string type = entry.text("type");
        if (type != "static") {
            const std::set<std::string> planned = {"modal", "harmonic", "transient", "nonlinear_static"};
            entry.fail("type", planned.contains(type)
                                   ? std::format("'{}' cases are not available in this build yet", type)
                                   : std::format("unknown case type '{}'", type));
        }
        StaticCase loadCase = readStaticCase(entry, study);
        if (!names.insert(loadCase.name).second) {
            entry.fail("name", std::format("another case is already named '{}'", loadCase.name));
        }
        entry.finish();
        study.cases.emplace_back(std::move(loadCase));
    }
    if (study.cases.empty()) {
        root.fail("cases", "at least one load case is required");
    }
}

// ---- Targets -------------------------------------------------------------------------------

constexpr std::array kQuantityNames = {
    std::pair{"mass", Quantity::Mass},
    std::pair{"mass_change", Quantity::MassChange},
    std::pair{"volume", Quantity::Volume},
    std::pair{"safety_factor", Quantity::SafetyFactor},
    std::pair{"stress", Quantity::Stress},
    std::pair{"displacement", Quantity::Displacement},
    std::pair{"compliance", Quantity::Compliance},
    std::pair{"natural_frequency", Quantity::NaturalFrequency},
    std::pair{"frequency_separation", Quantity::FrequencySeparation},
    std::pair{"harmonic_stress", Quantity::HarmonicStress},
    std::pair{"fatigue_life", Quantity::FatigueLife},
    std::pair{"peak_transient_stress", Quantity::PeakTransientStress},
    std::pair{"peak_acceleration", Quantity::PeakAcceleration},
    std::pair{"peak_contact_force", Quantity::PeakContactForce},
};

constexpr std::array kRelationNames = {
    std::pair{"<=", Relation::AtMost},   std::pair{">=", Relation::AtLeast},
    std::pair{"=", Relation::Equal},     std::pair{"minimize", Relation::Minimize},
    std::pair{"maximize", Relation::Maximize},
};

// Quantities that are measured per load case, as opposed to properties of the part.
bool isPerCase(Quantity quantity)
{
    return quantity != Quantity::Mass && quantity != Quantity::MassChange && quantity != Quantity::Volume;
}

// Quantities whose analyses arrive in later phases (docs/port-plan.md, section 8).
bool needsDynamicCase(Quantity quantity)
{
    switch (quantity) {
    case Quantity::NaturalFrequency:
    case Quantity::FrequencySeparation:
    case Quantity::HarmonicStress:
    case Quantity::FatigueLife:
    case Quantity::PeakTransientStress:
    case Quantity::PeakAcceleration:
    case Quantity::PeakContactForce:
        return true;
    default:
        return false;
    }
}

void readTargetValue(Fields& entry, Target& target)
{
    const bool needsValue = target.relation == Relation::AtMost || target.relation == Relation::AtLeast ||
                            target.relation == Relation::Equal;
    if (!needsValue) {
        if (entry.has("value")) {
            entry.fail("value", "minimize and maximize targets take no value");
        }
        return;
    }
    if (!entry.has("value")) {
        entry.fail("value", "is required for <=, >= and = targets");
    }
    const Json& raw = entry.raw("value");
    if (raw.is_string()) {
        // A percentage of the starting part, e.g. "-30%": only meaningful for mass change.
        const std::string text = raw.get<std::string>();
        double percent = 0.0;
        std::size_t used = 0;
        try {
            percent = std::stod(text, &used);
        } catch (const std::exception&) {
            used = 0;
        }
        if (used == 0 || used + 1 != text.size() || text.back() != '%') {
            entry.fail("value", std::format("'{}' is neither a number nor a percentage such as \"-30%\"", text));
        }
        if (target.quantity != Quantity::MassChange) {
            entry.fail("value", "a percentage is only meaningful for mass_change");
        }
        target.value = percent;
        target.valueIsPercentOfStart = true;
    } else {
        target.value = entry.number("value");
    }
}

Target readTarget(Fields entry, const Study& study)
{
    Target target;
    target.quantity = parseName(entry, "quantity", kQuantityNames);
    target.relation = parseName(entry, "relation", kRelationNames);
    readTargetValue(entry, target);

    if (needsDynamicCase(target.quantity)) {
        entry.fail("quantity", "this quantity needs modal, harmonic or transient cases, which are not available "
                               "in this build yet");
    }
    if (entry.has("cases")) {
        if (!isPerCase(target.quantity)) {
            entry.fail("cases", "this quantity is a property of the part, not of a load case");
        }
        target.cases = entry.textList("cases");
        for (const std::string& name : target.cases) {
            const bool exists = std::any_of(study.cases.begin(), study.cases.end(),
                                            [&](const LoadCase& c) { return caseName(c) == name; });
            if (!exists) {
                entry.fail("cases", std::format("no case named '{}'", name));
            }
        }
    }
    if (entry.has("at")) {
        target.at = entry.text("at");
        requireSelection(study, entry, "at", *target.at);
    }
    if (target.quantity == Quantity::Displacement && !target.at) {
        entry.fail("at", "a displacement target needs the selection it is measured at");
    }
    entry.finish();
    return target;
}

void readTargets(Fields& root, Study& study)
{
    for (Fields entry : root.objectList("targets")) {
        study.targets.push_back(readTarget(entry, study));
    }
    if (study.targets.empty()) {
        root.fail("targets", "at least one target is required");
    }
}

// ---- Start -----------------------------------------------------------------------------------

void readStart(Fields& root, Study& study, const std::filesystem::path& base)
{
    if (!root.has("start")) {
        return;  // Generate from the design space with full topological freedom.
    }
    Fields start = root.object("start");
    study.start.from = parseName(start, "from", std::array{std::pair{"design_space", StartFrom::DesignSpace},
                                                           std::pair{"part", StartFrom::Part},
                                                           std::pair{"previous_run", StartFrom::PreviousRun}});
    if (study.start.from == StartFrom::PreviousRun) {
        study.start.previousRun = resolve(base, start.text("run"));
    }
    if (start.has("freedom")) {
        study.start.freedom = parseName(start, "freedom", std::array{std::pair{"topology", Freedom::Topology},
                                                                     std::pair{"shape", Freedom::Shape}});
        if (study.start.from == StartFrom::DesignSpace && study.start.freedom == Freedom::Shape) {
            start.fail("freedom", "shape-only freedom needs an existing part to start from");
        }
    }
    start.finish();

    const bool massChangeTarget = std::any_of(study.targets.begin(), study.targets.end(), [](const Target& t) {
        return t.quantity == Quantity::MassChange;
    });
    if (massChangeTarget && study.start.from == StartFrom::DesignSpace) {
        root.fail("targets", "mass_change is measured against a starting part, but this study starts from the "
                             "design space");
    }
}

}  // namespace

Study parseStudy(std::string_view text, const std::filesystem::path& baseDirectory)
{
    Json document;
    try {
        document = Json::parse(text);
    } catch (const Json::parse_error& error) {
        throw InputError("(document)", std::format("invalid JSON: {}", error.what()));
    }
    Fields root = detail::rootFields(document);
    Study study;
    study.digest = sha256(text);
    readHeader(root, study);
    readPart(root, study, baseDirectory);
    if (root.has("frame")) {
        study.frame = readFrame(root.object("frame"));
    }
    readSelections(root, study);
    readDesignSpace(root, study, baseDirectory);
    readMaterials(root, study);
    readCases(root, study);
    readTargets(root, study);
    readStart(root, study, baseDirectory);
    root.finish();
    return study;
}

Study readStudy(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw InputError(path.string(), "cannot open the study file");
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    const std::filesystem::path absolute = std::filesystem::absolute(path).lexically_normal();
    try {
        Study study = parseStudy(contents.str(), absolute.parent_path());
        study.file = absolute;
        return study;
    } catch (const InputError& error) {
        // Prefix the file name so the message reads "study.json: cases[0].material: ...".
        const std::string fileName = path.filename().string();
        throw InputError(error.where().empty() ? fileName : fileName + ": " + error.where(), error.message());
    }
}

}  // namespace gso::study
