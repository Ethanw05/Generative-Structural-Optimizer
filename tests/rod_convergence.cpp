#include "gso/mechanics/equilibrium.hpp"
#include "test_support.hpp"
#include <iostream>
#include <numbers>

int main()
try
{
    using namespace gso;
    constexpr double length = 4., side = .4, forceN = 5., e = 2000., rho = 1200e-12;
    const double hz = .8 * std::sqrt(e / rho) / (4. * length);
    const double w = 2. * std::numbers::pi * hz;
    const std::complex<double> visco(1., w * 1.4e-7);
    const auto k = std::sqrt(std::complex<double>(w * w, -750. * w) * rho / (e * visco));
    const auto exact = forceN * forceN * std::tan(k * length) / (e * side * side * visco * k);
    double previous = 1.;
    for (unsigned nx : {6U, 10U, 16U, 24U})
    {
        const double h = length / nx;
        const int cross = std::max(2, static_cast<int>(std::ceil(side / h)));
        Model model{ImplicitSolid({{0, 0, 0}, {length, side, side}}, 1.),
                    {e, 0., 1200.},
                    {},
                    2.4 * h,
                    {{4 * nx, 6, 6}, 0, 20000},
                    {}};
        for (unsigned i = 0; i <= nx; ++i)
        {
            for (int j = 0; j <= cross; ++j)
            {
                for (int l = 0; l <= cross; ++l)
                {
                    Point x(length * (static_cast<double>(i) / nx), side * (static_cast<double>(j) / cross),
                            side * (static_cast<double>(l) / cross));
                    model.sites.push_back(x);
                    if (i == 0)
                    {
                        for (int a = 0; a < 3; ++a)
                        {
                            model.restraints.push_back({x, Point::Unit(a)});
                        }
                    }
                }
            }
        }
        std::vector<Force> traction;
        constexpr int panels = 8;
        for (int j = 0; j < panels; ++j)
        {
            for (int l = 0; l < panels; ++l)
            {
                for (double y : {-1., 1.})
                {
                    for (double z : {-1., 1.})
                    {
                        traction.push_back({{length, side * (j + .5 + y / std::sqrt(12.)) / panels,
                                             side * (l + .5 + z / std::sqrt(12.)) / panels},
                                            {forceN / (4. * panels * panels), 0, 0}});
                    }
                }
            }
        }
        auto system = assemble(model);
        auto response = solve(system, loadVector(model, system, traction), {hz, 750., 1.4e-7});
        const double error = std::abs(response.workNmm - exact) / std::abs(exact);
        std::cout << h << " mm: " << error << std::endl;
        if (nx >= 10)
        {
            check(error < .0005, "Refined wave solution exceeds the independent analytical error budget");
        }
        if (nx == 24)
        {
            check(error < previous, "Final fully three-dimensional refinement did not reduce the wave error");
        }
        previous = error;
    }
    check(previous < .0005, "Finest 3D response exceeds the analytical rod error threshold");
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
