#include "gso/mechanics/approximation.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <array>
#include <stdexcept>

namespace gso
{
namespace
{
using Four = Eigen::Vector4d;
using Moment = Eigen::Matrix4d;
struct Term
{
    int id;
    Four polynomial;
    double weight;
    Point slope;
};
} // namespace
std::vector<Shape> approximation(const std::vector<Point> &sites, const ImplicitSolid &solid, const Point &x,
                                 double support)
{
    if (!std::isfinite(support) || support <= 0.0 || !x.allFinite())
    {
        throw std::invalid_argument("Invalid displacement support or evaluation point");
    }
    if (!solid.contains(x))
    {
        throw std::runtime_error("An integration, load or restraint point has lost material support");
    }
    std::vector<Term> terms;
    Moment moment = Moment::Zero();
    std::array<Moment, 3> derivatives{Moment::Zero(), Moment::Zero(), Moment::Zero()};
    for (std::size_t i = 0; i < sites.size(); ++i)
    {
        const Point offset = (sites[i] - x) / support;
        const double r = offset.norm();
        if (r >= 1.0 || !solid.visible(x, sites[i]))
        {
            continue;
        }
        Four p;
        p << 1.0, offset;
        const double w = wendland(r);
        const Point dw = (20.0 * std::pow(1.0 - r, 3) / support) * offset;
        const Moment product = p * p.transpose();
        moment += w * product;
        for (int a = 0; a < 3; ++a)
        {
            derivatives[a] += dw[a] * product;
        }
        terms.push_back({static_cast<int>(i), p, w, dw});
    }
    Eigen::SelfAdjointEigenSolver<Moment> eigen(moment, Eigen::EigenvaluesOnly);
    if (terms.size() < 4 || eigen.info() != Eigen::Success ||
        eigen.eigenvalues()[0] <= 1e-11 * eigen.eigenvalues()[3])
    {
        throw std::runtime_error("Insufficient independent support in the visible solid");
    }
    const auto factor = moment.ldlt();
    const Four dual = factor.solve(Four::Unit(0));
    Eigen::Matrix<double, 4, 3> slope;
    for (int a = 0; a < 3; ++a)
    {
        slope.col(a) = factor.solve(Four::Unit(a + 1) / support - derivatives[a] * dual);
    }
    std::vector<Shape> shape;
    shape.reserve(terms.size());
    for (const auto &t : terms)
    {
        const double v = t.polynomial.dot(dual);
        shape.push_back({t.id, t.weight * v, v * t.slope + t.weight * slope.transpose() * t.polynomial});
    }
    return shape;
}
} // namespace gso
