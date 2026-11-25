#include "gso/mechanics/equilibrium.hpp"
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <numbers>
#include <stdexcept>

namespace gso
{
namespace
{
using Complex = std::complex<double>;
Sparse staticSaddle(const System &s, double scale)
{
    const int n = static_cast<int>(s.stiffness.rows());
    const int size = n + static_cast<int>(s.constraints.rows());
    Sparse matrix(size, size);
    std::vector<Eigen::Triplet<double>> entries;
    for (int c = 0; c < s.stiffness.outerSize(); ++c)
    {
        for (Sparse::InnerIterator it(s.stiffness, c); it; ++it)
        {
            entries.emplace_back(it.row(), it.col(), it.value() / scale);
        }
    }
    for (int c = 0; c < s.constraints.outerSize(); ++c)
    {
        for (Sparse::InnerIterator it(s.constraints, c); it; ++it)
        {
            entries.emplace_back(n + it.row(), it.col(), it.value());
            entries.emplace_back(it.col(), n + it.row(), it.value());
        }
    }
    matrix.setFromTriplets(entries.begin(), entries.end());
    matrix.makeCompressed();
    return matrix;
}
} // namespace
Response solve(const System &s, const Eigen::VectorXd &force, Excitation drive)
{
    if (!std::isfinite(drive.hertz) || drive.hertz < 0 || !std::isfinite(drive.massDampingPerSecond) ||
        drive.massDampingPerSecond < 0 || !std::isfinite(drive.stiffnessDampingSeconds) ||
        drive.stiffnessDampingSeconds < 0 || force.size() != s.stiffness.rows() || !force.allFinite() ||
        force.stableNorm() == 0.)
    {
        throw std::invalid_argument("Invalid excitation or applied force");
    }
    const int n = static_cast<int>(force.size());
    const double scale = s.stiffness.diagonal().cwiseAbs().maxCoeff();
    if (!std::isfinite(scale) || scale <= 0.)
    {
        throw std::runtime_error("Zero or nonfinite elastic stiffness");
    }
    const auto saddle = staticSaddle(s, scale);
    Eigen::SparseQR<Sparse, Eigen::COLAMDOrdering<int>> rankCheck;
    rankCheck.setPivotThreshold(1e-11);
    rankCheck.compute(saddle);
    if (rankCheck.info() != Eigen::Success || rankCheck.rank() != saddle.rows())
    {
        throw std::runtime_error("Singular restrained solid, dependent fixtures, or inadequate integration");
    }
    const double w = 2.0 * std::numbers::pi * drive.hertz;
    const Complex stiffness(1., w * drive.stiffnessDampingSeconds),
        mass(-w * w, w * drive.massDampingPerSecond);
    Eigen::SparseMatrix<Complex> a = saddle.cast<Complex>();
    for (int c = 0; c < n; ++c)
    {
        for (Eigen::SparseMatrix<Complex>::InnerIterator it(a, c); it; ++it)
        {
            if (it.row() < n)
            {
                it.valueRef() *= stiffness;
            }
        }
        for (Sparse::InnerIterator it(s.mass, c); it; ++it)
        {
            a.coeffRef(it.row(), it.col()) += mass * it.value() / scale;
        }
    }
    a.makeCompressed();
    Eigen::VectorXcd rhs = Eigen::VectorXcd::Zero(a.rows());
    rhs.head(n) = force.cast<Complex>() / scale;
    Eigen::SparseLU<Eigen::SparseMatrix<Complex>> solver;
    solver.compute(a);
    if (solver.info() != Eigen::Success)
    {
        throw std::runtime_error("Singular dynamic equilibrium system");
    }
    const Eigen::VectorXcd solution = solver.solve(rhs);
    if (solver.info() != Eigen::Success || !solution.allFinite())
    {
        throw std::runtime_error("Equilibrium solve failed");
    }
    const double residual = (a * solution - rhs).norm() / rhs.norm();
    if (!std::isfinite(residual) || residual > 1e-8)
    {
        throw std::runtime_error("Equilibrium residual exceeds tolerance");
    }
    const Eigen::VectorXcd displacement = solution.head(n);
    return {force.cast<Complex>().dot(displacement), displacement,
            -scale * solution.tail(s.constraints.rows()), residual};
}
} // namespace gso
