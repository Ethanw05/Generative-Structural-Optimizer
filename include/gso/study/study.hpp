#pragma once

// The in-memory study: everything the generator needs to know about one problem.
// It is read from a versioned JSON study file (gso/study/study_file.hpp); the format is
// described in docs/study-format.md.
//
// Internal units: millimetres, newtons, seconds, megapascals.

#include "gso/core/sha256.hpp"
#include "gso/study/material.hpp"

#include <Eigen/Core>

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace gso::study {

using Vec3 = Eigen::Vector3d;

// A right-handed frame for entering loads in the part's own axes.
// Local x lies along `xAxis`, local z along `upAxis`, local y = z cross x.
struct LocalFrame {
    Vec3 origin = Vec3::Zero();
    Vec3 xAxis = Vec3::UnitX();
    Vec3 upAxis = Vec3::UnitZ();
};

// A planar CAD face, found as the planar facet group of the part surface that contains
// `point` (within `toleranceMm`) and whose outward normal matches `normal`.
struct PlanarFaceSelection {
    Vec3 point = Vec3::Zero();
    Vec3 normal = Vec3::UnitZ();
    double toleranceMm = 0.01;
};

struct Selection {
    std::string name;
    PlanarFaceSelection planarFace;
};

enum class LoadFrame { Global, Local };

// A resultant force on a selection, distributed over it.
struct ForceLoad {
    std::string on;
    Vec3 forceN = Vec3::Zero();
    LoadFrame frame = LoadFrame::Global;
};

// A resultant moment on a selection about its centroid (e.g. motor torque on bolt faces).
struct TorqueLoad {
    std::string on;
    Vec3 torqueNmm = Vec3::Zero();
    LoadFrame frame = LoadFrame::Global;
};

using Load = std::variant<ForceLoad, TorqueLoad>;

struct StaticCase {
    std::string name;
    std::string material;
    std::vector<std::string> fixtures;
    std::vector<Load> loads;
};

// Further case types (modal, harmonic, transient, nonlinear) are added with the analyses
// that evaluate them; until then the reader rejects them with a clear message.
using LoadCase = std::variant<StaticCase>;

const std::string& caseName(const LoadCase& loadCase);

enum class Quantity {
    Mass,
    MassChange,
    Volume,
    SafetyFactor,
    Stress,
    Displacement,
    Compliance,
    NaturalFrequency,
    FrequencySeparation,
    HarmonicStress,
    FatigueLife,
    PeakTransientStress,
    PeakAcceleration,
    PeakContactForce,
};

enum class Relation { AtMost, AtLeast, Equal, Minimize, Maximize };

// One generator target (docs/port-plan.md, section 4). Targets with a value are
// constraints; minimize/maximize targets are objectives.
struct Target {
    Quantity quantity = Quantity::Mass;
    Relation relation = Relation::Minimize;
    std::optional<double> value;
    bool valueIsPercentOfStart = false;  // "-30%" for mass_change
    std::vector<std::string> cases;      // empty: every case the quantity applies to
    std::optional<std::string> at;       // a selection, for point-wise quantities
};

struct PreservedRegion {
    std::string selection;
    double depthMm = 0.0;  // material kept behind the face
};

// The region material may occupy: the part's bounding box, the part itself (material can
// only be removed), or an explicit axis-aligned box.
enum class Envelope { BoundingBox, Part, Box };

struct DesignSpace {
    Envelope envelope = Envelope::BoundingBox;
    Vec3 boxMinMm = Vec3::Zero();  // only for Envelope::Box
    Vec3 boxMaxMm = Vec3::Zero();
    std::vector<PreservedRegion> preserved;
    std::vector<std::filesystem::path> excludedSolids;  // keep-out solids (OBJ), absolute paths
};

enum class StartFrom { DesignSpace, Part, PreviousRun };
enum class Freedom { Topology, Shape };

// Where generation starts (docs/port-plan.md, "Evolving a part").
struct Start {
    StartFrom from = StartFrom::DesignSpace;
    std::filesystem::path previousRun;  // absolute; only for StartFrom::PreviousRun
    Freedom freedom = Freedom::Topology;
};

struct Study {
    int formatVersion = 0;
    std::filesystem::path file;  // absolute path of the study file
    Digest digest{};             // SHA-256 of the study file's bytes
    std::filesystem::path part;  // absolute path of the part (OBJ, SLDPRT or SLDASM)
    std::optional<Digest> partDigest;
    std::optional<LocalFrame> frame;
    std::map<std::string, Selection> selections;
    DesignSpace designSpace;
    std::map<std::string, Material> materials;
    std::vector<LoadCase> cases;
    std::vector<Target> targets;
    Start start;
};

}  // namespace gso::study
