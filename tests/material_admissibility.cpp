// Orthotropic constants are admissible only if the strain energy is positive definite.
// Lempriere (1968, AIAA J. 6(11):2226) gives the conditions in closed form: positive moduli,
// |nu_ij| < sqrt(E_i / E_j) for every pair, and
//     1 - nu12 nu21 - nu23 nu32 - nu13 nu31 - 2 nu21 nu32 nu13 > 0,  with nu_ji = nu_ij E_j / E_i.
// The study reader must accept and reject exactly what these conditions do.

#include "gso/core/error.hpp"
#include "gso/study/study_file.hpp"

#include <doctest/doctest.h>

#include <cmath>
#include <format>
#include <string>

namespace {

std::string studyWithOrthotropic(double nu12, double nu13, double nu23)
{
    return std::format(R"({{
      "format": "gso-study", "version": 1,
      "units": {{"length": "mm", "force": "N", "time": "s", "stress": "MPa", "mass": "kg"}},
      "part": {{"source": "part.obj"}},
      "selections": {{"a": {{"planar_face": {{"point": [0, 0, 0], "normal": [0, 0, -1]}}}},
                     "b": {{"planar_face": {{"point": [0, 0, 1], "normal": [0, 0, 1]}}}}}},
      "design_space": {{"envelope": "bounding_box", "preserve": [], "exclude": []}},
      "materials": {{"m": {{"density_kg_m3": 1500, "source": "test",
        "elasticity": {{"orthotropic": {{"e1_mpa": 1000, "e2_mpa": 1000, "e3_mpa": 1000,
          "nu12": {}, "nu13": {}, "nu23": {}, "g12_mpa": 400, "g13_mpa": 400, "g23_mpa": 400}}}}}}}},
      "cases": [{{"name": "c", "type": "static", "material": "m", "fixtures": ["a"],
                  "loads": [{{"on": "b", "force_n": [0, 0, 1]}}]}}],
      "targets": [{{"quantity": "compliance", "relation": "minimize"}}]
    }})", nu12, nu13, nu23);
}

// Lempriere's conditions for equal moduli (so nu_ji = nu_ij).
bool lempriereAdmissible(double nu12, double nu13, double nu23)
{
    const bool pairwise = std::abs(nu12) < 1.0 && std::abs(nu13) < 1.0 && std::abs(nu23) < 1.0;
    const double determinant = 1.0 - nu12 * nu12 - nu23 * nu23 - nu13 * nu13 - 2.0 * nu12 * nu23 * nu13;
    return pairwise && determinant > 0.0;
}

bool readerAccepts(double nu12, double nu13, double nu23)
{
    try {
        gso::study::parseStudy(studyWithOrthotropic(nu12, nu13, nu23), ".");
        return true;
    } catch (const gso::InputError&) {
        return false;
    }
}

}  // namespace

TEST_CASE("Orthotropic constants are accepted exactly when Lempriere's conditions hold")
{
    struct Constants {
        double nu12, nu13, nu23;
    };
    const Constants cases[] = {
        {0.30, 0.30, 0.30},   // ordinary material: admissible
        {0.90, 0.00, 0.00},   // large in-plane ratio alone: admissible (determinant 0.19)
        {0.90, 0.30, 0.30},   // pairwise bounds hold, determinant -0.152: inadmissible
        {1.10, 0.00, 0.00},   // violates |nu12| < sqrt(E1/E2)
        {-0.50, -0.50, -0.50} // auxetic, determinant 1 - 0.75 + 0.25 = 0.5: admissible
    };
    for (const Constants& c : cases) {
        INFO("nu12 = " << c.nu12 << ", nu13 = " << c.nu13 << ", nu23 = " << c.nu23);
        CHECK(readerAccepts(c.nu12, c.nu13, c.nu23) == lempriereAdmissible(c.nu12, c.nu13, c.nu23));
    }
}
