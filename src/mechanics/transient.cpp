#include "gso/mechanics/transient.hpp"
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
namespace gso
{
namespace
{
using Vec = Eigen::VectorXd;
double value(const std::vector<LoadKnot> &curve, double t)
{
    auto next = std::lower_bound(curve.begin(), curve.end(), t,
                                 [](const LoadKnot &k, double time) { return k.seconds < time; });
    if (next == curve.begin())
    {
        return next->multiplier;
    }
    if (next == curve.end())
    {
        return curve.back().multiplier;
    }
    const auto &prev = *(next - 1);
    return std::lerp(prev.multiplier, next->multiplier, (t - prev.seconds) / (next->seconds - prev.seconds));
}
std::vector<double> boundaries(const TransientCase &c, Eigen::Index n)
{
    if (!std::isfinite(c.endSeconds) || c.endSeconds <= 0 || !std::isfinite(c.maxStepSeconds) ||
        c.maxStepSeconds <= 0 || !std::isfinite(c.massDampingPerSecond) || c.massDampingPerSecond < 0 ||
        !std::isfinite(c.stiffnessDampingSeconds) || c.stiffnessDampingSeconds < 0)
    {
        throw std::invalid_argument("Invalid time integration settings");
    }
    std::set<double> knots{0., c.endSeconds};
    for (const auto &l : c.loads)
    {
        if (l.spatialForce.size() != n || !l.spatialForce.allFinite() || l.history.empty() ||
            l.history.front().seconds != 0.)
        {
            throw std::invalid_argument(
                "A time load requires a spatial force and a history starting at zero");
        }
        double previous = -1.;
        for (const auto &k : l.history)
        {
            if (!std::isfinite(k.seconds) || !std::isfinite(k.multiplier) || k.seconds <= previous ||
                k.seconds > c.endSeconds)
            {
                throw std::invalid_argument("Invalid load history knot");
            }
            knots.insert(k.seconds);
            previous = k.seconds;
        }
    }
    for (const auto &p : c.contacts)
    {
        if (p.direction.size() != n || !std::isfinite(p.direction.norm()) || p.direction.norm() == 0. ||
            !std::isfinite(p.initialGapMm) || p.initialGapMm < 0. || !std::isfinite(p.stiffnessNPerMm) ||
            p.stiffnessNPerMm <= 0.)
        {
            throw std::invalid_argument("Invalid normal contact");
        }
    }
    return {knots.begin(), knots.end()};
}
class RestrainedFactor
{
    Eigen::SparseLU<Sparse> solver;
    Sparse matrix;
    Eigen::Index dofs = 0;
    double scale = 1.;

  public:
    void factor(const Sparse &tangent, const Sparse &constraint)
    {
        dofs = tangent.rows();
        scale = tangent.diagonal().cwiseAbs().maxCoeff();
        if (!std::isfinite(scale) || scale <= 0.)
        {
            throw std::runtime_error("Singular transient operator");
        }
        std::vector<Eigen::Triplet<double>> entries;
        for (int j = 0; j < tangent.outerSize(); ++j)
        {
            for (Sparse::InnerIterator it(tangent, j); it; ++it)
            {
                entries.emplace_back(it.row(), it.col(), it.value() / scale);
            }
        }
        for (int j = 0; j < constraint.outerSize(); ++j)
        {
            for (Sparse::InnerIterator it(constraint, j); it; ++it)
            {
                entries.emplace_back(dofs + it.row(), it.col(), it.value());
                entries.emplace_back(it.col(), dofs + it.row(), it.value());
            }
        }
        matrix.resize(dofs + constraint.rows(), dofs + constraint.rows());
        matrix.setFromTriplets(entries.begin(), entries.end());
        solver.compute(matrix);
        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("Transient factorization failed");
        }
    }
    Vec apply(const Vec &f, double &maxResidual)
    {
        Vec rhs = Vec::Zero(matrix.rows());
        rhs.head(dofs) = f / scale;
        const Vec x = solver.solve(rhs);
        double r = (matrix * x - rhs).norm() / std::max(1e-30, rhs.norm());
        if (solver.info() != Eigen::Success || !x.allFinite() || !std::isfinite(r) || r > 1e-8)
        {
            throw std::runtime_error("Transient balance failed");
        }
        maxResidual = std::max(maxResidual, r);
        return x.head(dofs);
    }
};
} // namespace
TimeResponse integrateTimedHistory(const System &s, const TransientCase &c, const Vec &probe,
                                  const std::function<void(double, const Vec &)> &observe,
                                  const std::function<void(double, const Vec &)> &observeAcceleration)
{
    const auto n = s.stiffness.rows();
    const auto knots = boundaries(c, n);
    if (n <= 0 || s.stiffness.cols() != n || s.mass.rows() != n || s.mass.cols() != n ||
        s.constraints.cols() != n || c.initialVelocity.size() != n || !c.initialVelocity.allFinite() ||
        (s.constraints * c.initialVelocity).norm() > 1e-9 * std::max(1., c.initialVelocity.norm()) ||
        (probe.size() != 0 && (probe.size() != n || !probe.allFinite())))
    {
        throw std::invalid_argument("Invalid initial state or response probe");
    }
    const Sparse damping = c.massDampingPerSecond * s.mass + c.stiffnessDampingSeconds * s.stiffness;
    auto external = [&](double time) {
        Vec f = Vec::Zero(n);
        for (const auto &l : c.loads)
        {
            f += value(l.history, time) * l.spatialForce;
        }
        return f;
    };
    auto contactSet = [&](const Vec &u) {
        std::vector<bool> active;
        for (const auto &p : c.contacts)
        {
            active.push_back(p.direction.dot(u) + p.initialGapMm < 0.);
        }
        return active;
    };
    TimeResponse out;
    Vec u = Vec::Zero(n), v = c.initialVelocity;
    RestrainedFactor factor;
    factor.factor(s.mass, s.constraints);
    Vec a = factor.apply(external(0.) - damping * v, out.maxResidual);
    if (observeAcceleration) { observeAcceleration(0., a); }
    for (std::size_t k = 1; k < knots.size(); ++k)
    {
        const double interval = knots[k] - knots[k - 1], count = std::ceil(interval / c.maxStepSeconds);
        if (count > 1000000. - out.steps)
        {
            throw std::invalid_argument("Time integration step budget exceeded");
        }
        const auto steps = static_cast<unsigned>(count);
        const double h = interval / steps;
        const double b = h * h * .25, g = h * .5;
        const Sparse base = s.mass + g * damping + b * s.stiffness;
        std::vector<bool> cached;
        bool initialized = false;
        for (unsigned step = 1; step <= steps; ++step)
        {
            const Vec up = u + h * v + b * a, vp = v + g * a;
            const double time = step == steps ? knots[k] : knots[k - 1] + step * h;
            const Vec rhs = external(time) - s.stiffness * up -
                            damping * vp;
            auto active = contactSet(up);
            bool converged = false;
            for (unsigned it = 0; it < 32; ++it)
            {
                if (!initialized || cached != active)
                {
                    Sparse tangent = base;
                    for (std::size_t i = 0; i < active.size(); ++i)
                    {
                        if (active[i])
                        {
                            const auto &p = c.contacts[i];
                            tangent += b * p.stiffnessNPerMm * (p.direction * p.direction.transpose());
                        }
                    }
                    factor.factor(tangent, s.constraints);
                    cached = active;
                    initialized = true;
                }
                Vec effective = rhs;
                for (std::size_t i = 0; i < active.size(); ++i)
                {
                    if (active[i])
                    {
                        const auto &p = c.contacts[i];
                        effective -=
                            p.direction * (p.stiffnessNPerMm * (p.initialGapMm + p.direction.dot(up)));
                    }
                }
                a = factor.apply(effective, out.maxResidual);
                u = up + b * a;
                auto next = contactSet(u);
                if (active == next)
                {
                    converged = true;
                    break;
                }
                active = std::move(next);
            }
            if (!converged)
            {
                throw std::runtime_error("Contact iteration failed to settle");
            }
            v = vp + g * a;
            ++out.steps;
            if (observeAcceleration) { observeAcceleration(time, a); }
            if (observe)
            {
                observe(time, u);
            }
            out.peakStrainEnergyNmm = std::max(out.peakStrainEnergyNmm, .5 * u.dot(s.stiffness * u));
            double compressiveForce = 0.;
            for (const auto &p : c.contacts)
            {
                compressiveForce += p.stiffnessNPerMm * std::max(0., -p.initialGapMm - p.direction.dot(u));
            }
            out.peakContactForceN = std::max(out.peakContactForceN, compressiveForce);
            if (probe.size())
            {
                out.peakProbeMm = std::max(out.peakProbeMm, std::abs(probe.dot(u)));
            }
        }
    }
    out.displacement = std::move(u);
    out.velocity = std::move(v);
    return out;
}
TimeResponse integrateHistory(const System &s, const TransientCase &c, const Vec &probe,
                              const std::function<void(const Eigen::VectorXd &)> &observe)
{
    if (!observe) { return integrateTimedHistory(s, c, probe, {}); }
    return integrateTimedHistory(s, c, probe, [&](double, const Vec &u) { observe(u); });
}
} // namespace gso
