#include "gso/optimization/surface_gradient.hpp"
#include <Eigen/Geometry>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gso
{
Eigen::VectorXd surfaceMotion(std::span<const Eigen::Vector3d> points,
                              std::span<const DrivenTriangle> faces,
                              double moveBound, double availableVolume,
                              std::span<const std::array<double, 2>> bounds, double regularizationMm)
{
    if (points.empty() || faces.empty() || points.size() > std::numeric_limits<int>::max() ||
        !std::isfinite(regularizationMm) || regularizationMm < 0. ||
        !std::isfinite(moveBound) || moveBound <= 0. ||
        !std::isfinite(availableVolume) || availableVolume < 0.)
    {
        throw std::invalid_argument("Invalid surface motion problem");
    }
    const int count = static_cast<int>(points.size());
    std::vector<bool> locked(count, false);
    for (const auto& face : faces)
    {
        if (!std::isfinite(face.descent)) { throw std::invalid_argument("Nonfinite surface sensitivity"); }
        for (const int i : face.vertices)
        {
            if (i < 0 || i >= count) { throw std::invalid_argument("Invalid surface index"); }
            if (face.locked) { locked[i] = true; }
        }
    }
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(count, -moveBound);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(count, moveBound);
    if (!bounds.empty() && bounds.size() != points.size()) { throw std::invalid_argument("Missing CAD bounds"); }
    for (int i = 0; i < count; ++i)
    {
        if (!bounds.empty())
        {
            const auto [lo, hi] = bounds[i];
            if (!std::isfinite(lo) || !std::isfinite(hi) || lo > 0. || hi < 0.)
            {
                throw std::invalid_argument("Invalid CAD motion limits");
            }
            lower[i] = std::max(lo, -moveBound);
            upper[i] = std::min(hi, moveBound);
        }
        if (locked[i]) { lower[i] = upper[i] = 0.; }
    }
    Eigen::VectorXd area = Eigen::VectorXd::Zero(count);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(count);
    std::vector<Eigen::Triplet<double>> entries;
    for (const auto& face : faces)
    {
        const auto& v = face.vertices;
        const Eigen::Vector3d normal = (points[v[1]] - points[v[0]]).cross(points[v[2]] - points[v[0]]);
        const double twiceArea = normal.norm();
        if (!std::isfinite(twiceArea) || twiceArea <= 0.) { throw std::invalid_argument("Invalid surface triangle"); }
        std::array<Eigen::Vector3d, 3> gradients;
        for (int i = 0; i < 3; ++i)
        {
            const Eigen::Vector3d edge = points[v[(i + 2) % 3]] - points[v[(i + 1) % 3]];
            gradients[i] = (normal / twiceArea).cross(edge) / twiceArea;
            area[v[i]] += twiceArea / 6.;
            if (!locked[v[i]]) { rhs[v[i]] += (twiceArea / 6.) * face.descent; }
        }
        // Squared regularization length is the local triangle area. This resolves the
        // gradient on its analysis mesh without borrowing a length from a CAD feature.
        const double faceArea = twiceArea / 2.;
        const double lengthSquared = std::max(faceArea, regularizationMm * regularizationMm);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                if (!locked[v[i]] && !locked[v[j]])
                {
                    entries.emplace_back(v[i], v[j], faceArea * lengthSquared * gradients[i].dot(gradients[j]));
                }
            }
        }
    }
    Eigen::VectorXd volumeGradient = area;
    for (int i = 0; i < count; ++i)
    {
        if (area[i] <= 0.) { throw std::invalid_argument("Unused surface vertex"); }
        entries.emplace_back(i, i, area[i]);
        if (locked[i]) { volumeGradient[i] = 0.; }
    }
    Eigen::SparseMatrix<double> metric(count, count);
    metric.setFromTriplets(entries.begin(), entries.end());
    // Split the metric solve from projection onto the intersection of the local
    // move limits and available volume. A singular peak can saturate its own step.
    Eigen::SparseMatrix<double> system = metric;
    // Balance the mass and stiffness scales locally. Very slender analysis faces
    // can otherwise require thousands of splitting iterations at long wavelengths.
    const Eigen::VectorXd penalty = (area.array() * metric.diagonal().array()).sqrt();
    const Eigen::VectorXd volumeRatio = volumeGradient.cwiseQuotient(penalty);
    for (int i = 0; i < count; ++i) { system.coeffRef(i, i) += penalty[i]; }
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(system);
    if (solver.info() != Eigen::Success) { throw std::runtime_error("Surface metric factorization failed"); }
    Eigen::VectorXd bounded = Eigen::VectorXd::Zero(count);
    Eigen::VectorXd dual = bounded;
    const double tolerance = 1.e-6 * moveBound;
    for (int iteration = 0; iteration < 2000; ++iteration)
    {
        const Eigen::VectorXd smooth = solver.solve(moveBound * rhs + penalty.cwiseProduct(bounded - dual));
        if (!smooth.allFinite()) { throw std::runtime_error("Nonfinite surface update"); }
        const Eigen::VectorXd trial = smooth + dual;
        auto project = [&](double multiplier) -> Eigen::VectorXd {
            return (trial.array() - multiplier * volumeRatio.array()).max(lower.array()).min(upper.array()).matrix();
        };
        Eigen::VectorXd next = project(0.);
        if (volumeGradient.dot(next) > availableVolume)
        {
            double low = 0.;
            double high = 0.;
            for (int i = 0; i < count; ++i)
            {
                if (volumeRatio[i] > 0.) { high = std::max(high, (trial[i] - lower[i]) / volumeRatio[i]); }
            }
            for (int k = 0; k < 64 && (high - low) * volumeRatio.maxCoeff() > tolerance * .01; ++k)
            {
                const double middle = .5 * (low + high);
                if (volumeGradient.dot(project(middle)) > availableVolume) { low = middle; }
                else { high = middle; }
            }
            next = project(high);
        }
        const double difference = (next - bounded).lpNorm<Eigen::Infinity>();
        const Eigen::VectorXd residual = smooth - next;
        bounded = std::move(next);
        if (residual.lpNorm<Eigen::Infinity>() <= tolerance && difference <= tolerance) { return bounded; }
        dual += residual;
    }
    throw std::runtime_error("Constrained surface motion did not converge");
}
}
