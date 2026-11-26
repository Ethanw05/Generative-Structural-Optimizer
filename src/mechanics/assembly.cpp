#include "gso/mechanics/assembly.hpp"
#include "gso/mechanics/approximation.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gso
{
namespace
{
using Entry = Eigen::Triplet<double>;
using Elasticity = Eigen::Matrix<double, 6, 6>;
using Strain = Eigen::Matrix<double, 6, 3>;
Elasticity isotropic(Material m)
{
    if (!std::isfinite(m.modulusMPa) || m.modulusMPa <= 0.0 || !std::isfinite(m.poisson) ||
        m.poisson <= -1.0 || m.poisson >= 0.5 || !std::isfinite(m.densityKgM3) || m.densityKgM3 <= 0.0)
    {
        throw std::invalid_argument("Invalid isotropic elastic material");
    }
    const double shear = m.modulusMPa / (2.0 * (1.0 + m.poisson));
    const double lambda = 2.0 * shear * m.poisson / (1.0 - 2.0 * m.poisson);
    Elasticity c = Elasticity::Zero();
    c.topLeftCorner<3, 3>().setConstant(lambda);
    c.diagonal().head<3>().array() += 2.0 * shear;
    c.diagonal().tail<3>().setConstant(shear);
    return c;
}
Strain strain(const Point &g)
{
    Strain b;
    b << g.x(), 0, 0, 0, g.y(), 0, 0, 0, g.z(), g.y(), g.x(), 0, 0, g.z(), g.y(), g.z(), 0, g.x();
    return b;
}
} // namespace
System assemble(const Model &model)
{
    const Elasticity c = isotropic(model.material);
    System sys;
    for (const auto &x : model.sites)
    {
        if (!x.allFinite())
        {
            throw std::invalid_argument("Nonfinite displacement site");
        }
        if (model.solid.contains(x))
        {
            sys.sites.push_back(x);
        }
    }
    if (model.sites.size() < 4 ||
        model.sites.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 6) ||
        model.restraints.size() > 3 * model.sites.size())
    {
        throw std::invalid_argument("Invalid displacement site or restraint count");
    }
    if (sys.sites.size() < 4 || model.restraints.size() > 3 * sys.sites.size())
    {
        throw std::runtime_error("Candidate removed displacement support required by the fixtures");
    }
    const int n = 3 * static_cast<int>(sys.sites.size());
    sys.stiffness.resize(n, n);
    sys.mass.resize(n, n);
    sys.constraints.resize(static_cast<int>(model.restraints.size()), n);
    std::vector<Entry> boundary;
    for (std::size_t r = 0; r < model.restraints.size(); ++r)
    {
        const auto &bc = model.restraints[r];
        if (!bc.direction.allFinite() || std::abs(bc.direction.norm() - 1.0) > 1e-8)
        {
            throw std::invalid_argument("Restraint direction must be a unit vector");
        }
        const auto basis = approximation(sys.sites, model.solid, bc.location, model.support);
        for (const auto &s : basis)
        {
            for (int a = 0; a < 3; ++a)
            {
                boundary.emplace_back(static_cast<int>(r), 3 * s.site + a, s.value * bc.direction[a]);
            }
        }
    }
    sys.constraints.setFromTriplets(boundary.begin(), boundary.end());
    const auto volume = integrate(model.solid, model.integration);
    sys.unresolvedVolume = volume.unresolvedVolume;
    sys.lowerMassGrams = volume.interiorVolume * model.material.densityKgM3 * 1e-6;
    if (volume.points.empty())
    {
        throw std::runtime_error("Integrated solid has zero sampled volume");
    }
    // Inserting into a compressed matrix one coefficient at a time shifts its storage
    // on each new entry. Overlapping supports repeat every coefficient many times, so
    // contributions are summed in bounded batches instead of one unbounded list.
    constexpr std::size_t batch = std::size_t{1} << 22;
    std::vector<Entry> stiffness, mass;
    const auto merge = [n](std::vector<Entry> &entries, Sparse &target) {
        Sparse part(n, n);
        part.setFromTriplets(entries.begin(), entries.end());
        target += part;
        entries.clear();
    };
    for (const auto &q : volume.points)
    {
        auto basis = approximation(sys.sites, model.solid, q.location, model.support);
        std::vector<Strain> b;
        for (const auto &s : basis)
        {
            b.push_back(strain(s.gradient));
        }
        const double grams = q.volume * model.material.densityKgM3 * 1e-6;
        sys.massGrams += grams;
        for (std::size_t i = 0; i < basis.size(); ++i)
        {
            const auto left = (q.volume * b[i].transpose() * c).eval();
            for (std::size_t j = 0; j <= i; ++j)
            {
                const Eigen::Matrix3d block = left * b[j];
                const double inertia = grams * 1e-6 * basis[i].value * basis[j].value;
                const int row = 3 * basis[i].site, col = 3 * basis[j].site;
                for (int a = 0; a < 3; ++a)
                {
                    mass.emplace_back(row + a, col + a, inertia);
                    if (i != j)
                    {
                        mass.emplace_back(col + a, row + a, inertia);
                    }
                    for (int d = 0; d < 3; ++d)
                    {
                        stiffness.emplace_back(row + a, col + d, block(a, d));
                        if (i != j)
                        {
                            stiffness.emplace_back(col + d, row + a, block(a, d));
                        }
                    }
                }
            }
        }
        if (stiffness.size() >= batch)
        {
            merge(stiffness, sys.stiffness);
            merge(mass, sys.mass);
        }
    }
    merge(stiffness, sys.stiffness);
    merge(mass, sys.mass);
    sys.stiffness.makeCompressed();
    sys.mass.makeCompressed();
    sys.constraints.makeCompressed();
    return sys;
}
} // namespace gso
