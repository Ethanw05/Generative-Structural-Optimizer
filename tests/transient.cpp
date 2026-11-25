#include "gso/mechanics/transient.hpp"
#include "gso/mechanics/stress.hpp"
#include "gso/mechanics/wrench.hpp"
#include "test_support.hpp"
#include <Eigen/Geometry>
#include <numbers>
int main()
{
    using namespace gso;
    const Wrench wrench{{4, 2, -1}, {1, -3, 7}, {6, 4, -12}};
    const auto loads = distribute(wrench, {{{0, 0, 0}, 1.}, {{1, 0, 0}, 2.}, {{0, 2, 1}, 3.}});
    Point force = Point::Zero(), moment = Point::Zero();
    for (const auto &f : loads)
    {
        force += f.newtons;
        moment += (f.location - wrench.originMm).cross(f.newtons);
    }
    near((force - wrench.forceN).norm(), 0., 1e-10, "Unbalanced force");
    near((moment - wrench.momentNmm).norm(), 0., 1e-10, "Unbalanced moment");
    rejects(
        [&] {
            distribute({Point::Zero(), Point::Zero(), Point::UnitX()},
                       {{Point::Zero(), 1.}, {Point::UnitX(), 1.}});
        },
        "Axial moment on collinear points");
    System s;
    s.mass.resize(1, 1);
    s.mass.insert(0, 0) = 1.;
    s.stiffness.resize(1, 1);
    s.stiffness.insert(0, 0) = 4.;
    s.constraints.resize(0, 1);
    const auto unit = Eigen::VectorXd::Ones(1).eval();
    TransientCase history{std::numbers::pi / 2., .002, Eigen::VectorXd::Zero(1), {{unit, {{0., 1.}}}}, {}};
    auto response = integrateHistory(s, history, unit);
    near(response.peakProbeMm, .5, 3e-6, "Step load dynamic amplification");
    history.loads.push_back({-unit, {{0., 1.}}});
    response = integrateHistory(s, history, unit);
    near(response.peakProbeMm, 0., 1e-12, "Simultaneous opposing loads do not cancel");
    s.stiffness.coeffRef(0, 0) = 0.;
    Eigen::SparseVector<double> dir(1);
    dir.insert(0) = 1.;
    history = {.25 + std::numbers::pi / 4., .0005, -2. * unit, {}, {{dir, .5, 16.}}};
    response = integrateHistory(s, history, unit);
    near(response.peakContactForceN, 8., .001, "Normal impact impulse/contact amplitude");
    near(response.velocity[0], 2., .001, "Elastic rebound after separation");
    Model m{ImplicitSolid({{0, 0, 0}, {1, 1, 1}}, 1.),
            {2000., .25, 1200.},
            {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}},
            5.,
            {{2, 2, 2}, 0, 1000},
            {}};
    const auto sys = assemble(m);
    StressRecovery stress(m, sys);
    Eigen::VectorXd u(12);
    for (int i = 0; i < 4; ++i)
    {
        const auto &x = m.sites[i];
        u.segment<3>(3 * i) = Point(.01 * x.y(), 0, 0);
    }
    near(stress.measure(u, 8.).aggregateMPa, 8. * std::sqrt(3.), 1e-9, "Uniform shear von Mises");
    Eigen::VectorXcd phasor = u.cast<std::complex<double>>() * std::complex<double>(0., 1.);
    near(stress.measureHarmonic(phasor, 8.).peakMPa, 8. * std::sqrt(3.), 1.e-9, "Harmonic stress phase");
    for (int i = 0; i < 4; ++i)
    {
        u.segment<3>(3 * i) = .01 * m.sites[i];
    }
    near(stress.measure(u, 8.).peakMPa, 0., 1e-10, "Hydrostatic field has deviatoric stress");
}
