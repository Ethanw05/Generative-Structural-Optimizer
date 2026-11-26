#include "study/material_reader.hpp"

#include <Eigen/Eigenvalues>

#include <format>

namespace gso::study {

bool isIsotropic(const Material& material)
{
    return std::holds_alternative<IsotropicElasticity>(material.elasticity);
}

namespace detail {
namespace {

IsotropicElasticity readIsotropic(Fields fields)
{
    IsotropicElasticity elastic;
    elastic.youngMpa = fields.positive("young_mpa");
    elastic.poisson = fields.number("poisson");
    // The strain energy is positive definite only for -1 < nu < 1/2.
    if (elastic.poisson <= -1.0 || elastic.poisson >= 0.5) {
        fields.fail("poisson", std::format("must lie strictly between -1 and 0.5, found {}", elastic.poisson));
    }
    fields.finish();
    return elastic;
}

OrthotropicElasticity readOrthotropic(Fields fields)
{
    OrthotropicElasticity elastic;
    elastic.e1Mpa = fields.positive("e1_mpa");
    elastic.e2Mpa = fields.positive("e2_mpa");
    elastic.e3Mpa = fields.positive("e3_mpa");
    elastic.nu12 = fields.number("nu12");
    elastic.nu13 = fields.number("nu13");
    elastic.nu23 = fields.number("nu23");
    elastic.g12Mpa = fields.positive("g12_mpa");
    elastic.g13Mpa = fields.positive("g13_mpa");
    elastic.g23Mpa = fields.positive("g23_mpa");
    fields.finish();
    return elastic;
}

// Transversely isotropic about axis 3 (the build direction of a printed part, or the
// fibre direction rotated to axis 3 for a unidirectional composite).
OrthotropicElasticity readTransverselyIsotropic(Fields fields)
{
    const double inPlaneYoung = fields.positive("in_plane_young_mpa");
    const double inPlanePoisson = fields.number("in_plane_poisson");
    const double axialYoung = fields.positive("axial_young_mpa");
    const double inPlaneToAxialPoisson = fields.number("in_plane_to_axial_poisson");
    const double axialShear = fields.positive("axial_shear_mpa");
    fields.finish();

    OrthotropicElasticity elastic;
    elastic.e1Mpa = inPlaneYoung;
    elastic.e2Mpa = inPlaneYoung;
    elastic.e3Mpa = axialYoung;
    elastic.nu12 = inPlanePoisson;
    elastic.nu13 = inPlaneToAxialPoisson;
    elastic.nu23 = inPlaneToAxialPoisson;
    // In the plane of isotropy the shear modulus follows from E and nu.
    elastic.g12Mpa = inPlaneYoung / (2.0 * (1.0 + inPlanePoisson));
    elastic.g13Mpa = axialShear;
    elastic.g23Mpa = axialShear;
    return elastic;
}

// The compliance matrix must be positive definite for the strain energy to be positive.
// Shear moduli were already checked positive, so only the normal-stress block remains.
void requirePositiveDefinite(const OrthotropicElasticity& e, const Fields& fields)
{
    Eigen::Matrix3d compliance;
    compliance << 1.0 / e.e1Mpa, -e.nu12 / e.e1Mpa, -e.nu13 / e.e1Mpa,
                  -e.nu12 / e.e1Mpa, 1.0 / e.e2Mpa, -e.nu23 / e.e2Mpa,
                  -e.nu13 / e.e1Mpa, -e.nu23 / e.e2Mpa, 1.0 / e.e3Mpa;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(compliance, Eigen::EigenvaluesOnly);
    if (solver.eigenvalues().minCoeff() <= 0.0) {
        fields.fail("", "these elastic constants are not physically admissible: the compliance matrix is not "
                        "positive definite (check the Poisson ratios against the moduli)");
    }
}

Elasticity readElasticity(Fields& material)
{
    auto [kind, fields] = material.singleVariant("elasticity");
    if (kind == "isotropic") {
        return readIsotropic(fields);
    }
    OrthotropicElasticity elastic;
    if (kind == "orthotropic") {
        elastic = readOrthotropic(fields);
    } else if (kind == "transversely_isotropic") {
        elastic = readTransverselyIsotropic(fields);
    } else {
        fields.fail("", std::format("unknown elasticity '{}'; expected isotropic, transversely_isotropic or "
                                    "orthotropic", kind));
    }
    requirePositiveDefinite(elastic, fields);
    return elastic;
}

StrengthCriterion readStrength(Fields& material)
{
    auto [kind, fields] = material.singleVariant("strength");
    StrengthCriterion criterion;
    if (kind == "von_mises") {
        criterion = VonMisesStrength{fields.positive("yield_mpa")};
    } else if (kind == "tresca") {
        criterion = TrescaStrength{fields.positive("yield_mpa")};
    } else if (kind == "max_principal") {
        criterion = MaxPrincipalStrength{fields.positive("tensile_mpa"), fields.positive("compressive_mpa")};
    } else if (kind == "tsai_wu") {
        TsaiWuStrength s;
        s.xtMpa = fields.positive("xt_mpa");
        s.xcMpa = fields.positive("xc_mpa");
        s.ytMpa = fields.positive("yt_mpa");
        s.ycMpa = fields.positive("yc_mpa");
        s.ztMpa = fields.positive("zt_mpa");
        s.zcMpa = fields.positive("zc_mpa");
        s.s12Mpa = fields.positive("s12_mpa");
        s.s13Mpa = fields.positive("s13_mpa");
        s.s23Mpa = fields.positive("s23_mpa");
        criterion = s;
    } else {
        fields.fail("", std::format("unknown strength criterion '{}'; expected von_mises, tresca, max_principal "
                                    "or tsai_wu", kind));
    }
    fields.finish();
    return criterion;
}

MeanStressCorrection parseCorrection(Fields& fields)
{
    const std::string name = fields.text("mean_stress_correction");
    if (name == "none") {
        return MeanStressCorrection::None;
    }
    if (name == "goodman") {
        return MeanStressCorrection::Goodman;
    }
    if (name == "gerber") {
        return MeanStressCorrection::Gerber;
    }
    fields.fail("mean_stress_correction", std::format("unknown correction '{}'; expected none, goodman or gerber", name));
}

BasquinFatigue readFatigue(Fields& material)
{
    auto [kind, fields] = material.singleVariant("fatigue");
    if (kind != "basquin") {
        fields.fail("", std::format("unknown fatigue model '{}'; expected basquin", kind));
    }
    BasquinFatigue fatigue;
    fatigue.coefficientMpa = fields.positive("coefficient_mpa");
    fatigue.exponent = fields.number("exponent");
    if (fatigue.exponent >= 0.0) {
        fields.fail("exponent", "must be negative: strength falls as the number of cycles grows");
    }
    fatigue.meanStressCorrection = parseCorrection(fields);
    fatigue.ultimateMpa = fields.optionalNumber("ultimate_mpa");
    if (fatigue.ultimateMpa && *fatigue.ultimateMpa <= 0.0) {
        fields.fail("ultimate_mpa", "must be positive");
    }
    if (fatigue.meanStressCorrection != MeanStressCorrection::None && !fatigue.ultimateMpa) {
        fields.fail("ultimate_mpa", "is required by the Goodman and Gerber mean-stress corrections");
    }
    fields.finish();
    return fatigue;
}

J2Plasticity readPlasticity(Fields& material)
{
    auto [kind, fields] = material.singleVariant("plasticity");
    if (kind != "j2") {
        fields.fail("", std::format("unknown plasticity model '{}'; expected j2", kind));
    }
    J2Plasticity plasticity{fields.positive("yield_mpa"), fields.nonNegative("hardening_mpa")};
    fields.finish();
    return plasticity;
}

}  // namespace

Material readMaterial(const std::string& name, Fields fields)
{
    Material material;
    material.name = name;
    material.densityKgPerM3 = fields.positive("density_kg_m3");
    material.elasticity = readElasticity(fields);
    if (fields.has("strength")) {
        material.strength = readStrength(fields);
    }
    if (fields.has("fatigue")) {
        material.fatigue = readFatigue(fields);
    }
    if (fields.has("plasticity")) {
        material.plasticity = readPlasticity(fields);
    }
    if (fields.has("damping_loss_factor")) {
        material.dampingLossFactor = fields.nonNegative("damping_loss_factor");
    }
    material.source = fields.text("source");
    if (material.source.empty()) {
        fields.fail("source", "must say where the values come from (data sheet, paper, test)");
    }
    fields.finish();
    return material;
}

}  // namespace detail
}  // namespace gso::study
