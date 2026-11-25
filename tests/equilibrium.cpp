#include "gso/mechanics/equilibrium.hpp"
#include "test_support.hpp"
#include <numbers>

int main()
{
    using namespace gso;
    Model model{ImplicitSolid({{0, 0, 0}, {4, 1, 1}}, 1.),
                {2000., 0., 1200.},
                {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {4, 0, 0}},
                9.,
                {{2, 2, 2}, 2, 10000},
                {}};
    for (int i = 0; i < 3; ++i)
    {
        for (int a = 0; a < 3; ++a)
        {
            model.restraints.push_back({model.sites[i], Point::Unit(a)});
        }
    }
    const auto system = assemble(model);
    const auto force = loadVector(model, system, {{{4, .5, .5}, {5, 0, 0}}});
    near(system.massGrams, .0048, 1e-14, "Physical mass");
    const auto stat = solve(system, force, {});
    near(stat.workNmm.real(), .05, 1e-11, "Uniform axial extension");
    near(stat.workNmm.imag(), 0., 1e-13, "Imaginary component of static work");
    near(stat.reactions[0].real() + stat.reactions[3].real() + stat.reactions[6].real(), -5., 1e-10,
         "Resultant support reaction");
    const double k = 500., m = 4. * 1200e-12 / 3.;
    for (double frequency : {2300., std::sqrt(k / m) / (2. * std::numbers::pi), 125000.})
    {
        const double w = 2. * std::numbers::pi * frequency;
        const auto result = solve(system, force, {frequency, 750., 1.4e-7});
        const auto exact = 25. / std::complex<double>(k - m * w * w, w * (750. * m + 1.4e-7 * k));
        check(std::abs(result.workNmm - exact) < 1e-9 * std::abs(exact), "Consistent-mass harmonic response");
    }
    model.restraints[7] = model.restraints[0];
    auto singular = assemble(model);
    rejects([&] { solve(singular, force, {}); }, "Dependent restraints accepted");
}
