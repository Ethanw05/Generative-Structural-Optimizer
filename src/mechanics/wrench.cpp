#include "gso/mechanics/wrench.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <stdexcept>
namespace gso
{
std::vector<Force> distribute(const Wrench &w, const std::vector<AttachmentPoint> &points)
{
    if (points.empty() || !w.originMm.allFinite() || !w.forceN.allFinite() || !w.momentNmm.allFinite())
    {
        throw std::invalid_argument("Invalid wrench or missing attachment");
    }
    Point center = Point::Zero();
    double total = 0.;
    for (const auto &p : points)
    {
        if (!p.position.allFinite() || !std::isfinite(p.weight) || p.weight <= 0.)
        {
            throw std::invalid_argument("Invalid attachment point or weight");
        }
        center += p.weight * (p.position - w.originMm);
        total += p.weight;
    }
    center /= total;
    const Point requested = w.momentNmm - center.cross(w.forceN);
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
    for (const auto &p : points)
    {
        const Point r = (p.position - w.originMm) - center;
        inertia += (p.weight / total) * (r.squaredNorm() * Eigen::Matrix3d::Identity() - r * r.transpose());
    }
    if (!inertia.allFinite() || !requested.allFinite())
    {
        throw std::invalid_argument("Nonfinite attachment geometry");
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(inertia);
    if (eig.info() != Eigen::Success)
    {
        throw std::runtime_error("Attachment decomposition failed");
    }
    Point omega = Point::Zero();
    for (int j = 0; j < 3; ++j)
    {
        if (eig.eigenvalues()[j] > 1e-12 * eig.eigenvalues().maxCoeff())
        {
            const auto axis = eig.eigenvectors().col(j);
            omega += axis * (axis.dot(requested) / eig.eigenvalues()[j]);
        }
    }
    if ((inertia * omega - requested).norm() > 1e-10 * std::max(1e-12, requested.norm()))
    {
        throw std::invalid_argument("Selected attachment cannot transmit the requested moment");
    }
    std::vector<Force> result;
    for (const auto &p : points)
    {
        result.push_back(
            {p.position, (p.weight / total) * (w.forceN + omega.cross((p.position - w.originMm) - center))});
    }
    return result;
}
} // namespace gso
