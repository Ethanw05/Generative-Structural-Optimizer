#pragma once

// Materials of any type: metals, composites, plastics (docs/port-plan.md, section 3).
// A material is data with a recorded source, made of four independent parts: elastic
// symmetry, strength criterion, fatigue behaviour and inelastic behaviour.
//
// Units: MPa for moduli and strengths, kg/m^3 for density.

#include <optional>
#include <string>
#include <variant>

namespace gso::study {

struct IsotropicElasticity {
    double youngMpa = 0.0;
    double poisson = 0.0;
};

// Axes 1, 2, 3 are the material axes: for a printed part 1 and 2 lie in the layer and
// 3 is the build direction; for a unidirectional composite 1 is the fibre direction.
// Transversely isotropic materials are stored in this form as well.
struct OrthotropicElasticity {
    double e1Mpa = 0.0, e2Mpa = 0.0, e3Mpa = 0.0;
    double nu12 = 0.0, nu13 = 0.0, nu23 = 0.0;  // nu_ij: contraction along j under stress along i
    double g12Mpa = 0.0, g13Mpa = 0.0, g23Mpa = 0.0;
};

using Elasticity = std::variant<IsotropicElasticity, OrthotropicElasticity>;

// Ductile metals and polymers.
struct VonMisesStrength {
    double yieldMpa = 0.0;
};

struct TrescaStrength {
    double yieldMpa = 0.0;
};

// Brittle materials (cast iron, ceramics, some resins).
struct MaxPrincipalStrength {
    double tensileMpa = 0.0;
    double compressiveMpa = 0.0;  // magnitude
};

// Anisotropic materials with unequal tension and compression strengths: composites and
// printed parts, which are weak across layers. Strengths are magnitudes along the
// material axes; s_ij are the shear strengths.
struct TsaiWuStrength {
    double xtMpa = 0.0, xcMpa = 0.0;  // axis 1
    double ytMpa = 0.0, ycMpa = 0.0;  // axis 2
    double ztMpa = 0.0, zcMpa = 0.0;  // axis 3
    double s12Mpa = 0.0, s13Mpa = 0.0, s23Mpa = 0.0;
};

using StrengthCriterion = std::variant<VonMisesStrength, TrescaStrength, MaxPrincipalStrength, TsaiWuStrength>;

enum class MeanStressCorrection { None, Goodman, Gerber };

// Stress amplitude S at N cycles to failure: S = coefficient * N^exponent (exponent < 0).
struct BasquinFatigue {
    double coefficientMpa = 0.0;
    double exponent = 0.0;
    MeanStressCorrection meanStressCorrection = MeanStressCorrection::None;
    std::optional<double> ultimateMpa;  // required by the Goodman and Gerber corrections
};

// Rate-independent J2 plasticity with linear isotropic hardening.
struct J2Plasticity {
    double yieldMpa = 0.0;
    double hardeningMpa = 0.0;
};

struct Material {
    std::string name;
    double densityKgPerM3 = 0.0;
    Elasticity elasticity;
    std::optional<StrengthCriterion> strength;
    std::optional<BasquinFatigue> fatigue;
    std::optional<J2Plasticity> plasticity;
    // Structural damping loss factor eta (about twice the damping ratio at resonance).
    std::optional<double> dampingLossFactor;
    std::string source;  // where the values come from: a data sheet, a paper, a test
};

bool isIsotropic(const Material& material);

}  // namespace gso::study
