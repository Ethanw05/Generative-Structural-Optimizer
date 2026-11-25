#include "gso/mechanics/boundary.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>
namespace gso
{
namespace
{
double area(const SurfacePatch &p)
{
    const double measure = p.edgeU.cross(p.edgeV).norm();
    if (!p.origin.allFinite() || !p.edgeU.allFinite() || !p.edgeV.allFinite() || !std::isfinite(measure) ||
        measure <= 1e-10 * p.edgeU.norm() * p.edgeV.norm())
    {
        throw std::invalid_argument("Degenerate surface patch");
    }
    return measure;
}
} // namespace
std::vector<Restraint> collocateRestraints(const std::vector<Point> &sites, const SurfacePatch &p,
                                           const std::vector<Point> &directions)
{
    const double measure = area(p);
    if (directions.empty())
    {
        throw std::invalid_argument("No restraint directions on a surface patch");
    }
    const Point normal = p.edgeU.cross(p.edgeV) / measure;
    Eigen::Matrix<double, 3, 2> edges;
    edges.col(0) = p.edgeU;
    edges.col(1) = p.edgeV;
    const Eigen::Matrix2d gram = edges.transpose() * edges;
    const auto factor = gram.ldlt();
    const double tolerance = 1e-9 * std::max(p.edgeU.norm(), p.edgeV.norm());
    std::vector<Restraint> result;
    for (const auto &site : sites)
    {
        const Point d = site - p.origin;
        if (std::abs(normal.dot(d)) > tolerance)
        {
            continue;
        }
        const Eigen::Vector2d uv = factor.solve(edges.transpose() * d);
        if (uv.minCoeff() < -1e-9 || uv.maxCoeff() > 1. + 1e-9)
        {
            continue;
        }
        for (const auto &direction : directions)
        {
            result.push_back({site, direction});
        }
    }
    if (result.empty())
    {
        throw std::invalid_argument("No boundary sites on the restrained surface patch");
    }
    return result;
}
std::vector<Force> integrateTraction(const SurfacePatch &p, const Point &traction, double maxPanel)
{
    const double measure = area(p);
    if (!traction.allFinite() || !std::isfinite(maxPanel) || maxPanel <= 0.)
    {
        throw std::invalid_argument("Invalid surface traction");
    }
    const double nu = std::max(1., std::ceil(p.edgeU.norm() / maxPanel)),
                 nv = std::max(1., std::ceil(p.edgeV.norm() / maxPanel));
    if (!std::isfinite(nu * nv) || nu * nv > 1000000.)
    {
        throw std::invalid_argument("Surface quadrature budget exceeded");
    }
    const int uCount = static_cast<int>(nu), vCount = static_cast<int>(nv);
    std::vector<Force> result;
    const Point force = traction * (measure / (4. * nu * nv));
    for (int i = 0; i < uCount; ++i)
    {
        for (int j = 0; j < vCount; ++j)
        {
            for (double a : {-1., 1.})
            {
                for (double b : {-1., 1.})
                {
                    const double u = (i + .5 + a / std::sqrt(12.)) / nu,
                                 v = (j + .5 + b / std::sqrt(12.)) / nv;
                    result.push_back({p.origin + u * p.edgeU + v * p.edgeV, force});
                }
            }
        }
    }
    return result;
}
} // namespace gso
