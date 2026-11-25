#include "gso/optimization/refinement.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gso
{
namespace
{
bool positive(double x)
{
    return std::isfinite(x) && x > 0.;
}
} // namespace
Decision assess(const std::vector<LevelResult> &levels, const Criteria &c)
{
    if (!positive(c.relativeMass) || !positive(c.relativeResponse) || c.responseLimits.empty())
    {
        throw std::invalid_argument("Invalid refinement criteria");
    }
    for (const auto &[name, limit] : c.responseLimits)
    {
        if (name.empty() || !positive(limit))
        {
            throw std::invalid_argument("Invalid named response limit");
        }
    }
    double previous = std::numeric_limits<double>::infinity();
    for (const auto &level : levels)
    {
        if (!positive(level.spacingMm) || level.spacingMm >= previous || !positive(level.massGrams) ||
            !std::isfinite(level.unresolvedMassGrams) || level.unresolvedMassGrams < 0. ||
            level.responses.size() != c.responseLimits.size())
        {
            throw std::invalid_argument("Inconsistent discrete analysis level");
        }
        const double roundoff = 1e-10 * level.massGrams;
        if (!std::isfinite(level.lowerMassGrams) || level.lowerMassGrams < 0. ||
            level.massGrams < level.lowerMassGrams - roundoff ||
            level.massGrams > level.lowerMassGrams + level.unresolvedMassGrams + roundoff)
        {
            throw std::invalid_argument("Mass is outside its geometric enclosure");
        }
        previous = level.spacingMm;
        for (const auto &[name, work] : level.responses)
        {
            if (!c.responseLimits.contains(name) || !std::isfinite(std::abs(work)))
            {
                throw std::invalid_argument("Invalid response identity or value");
            }
        }
    }
    if (levels.size() < 3)
    {
        return Decision::refine;
    }
    bool exceeded = false;
    for (std::size_t level = levels.size() - 3; level < levels.size(); ++level)
    {
        const auto &fine = levels[level];
        if (fine.unresolvedMassGrams > c.relativeMass * fine.massGrams)
        {
            return Decision::refine;
        }
        for (const auto &[name, limit] : c.responseLimits)
        {
            exceeded |= std::abs(fine.responses.at(name)) > limit;
        }
        if (level == levels.size() - 3)
        {
            continue;
        }
        const auto &coarse = levels[level - 1];
        if (std::abs(fine.massGrams - coarse.massGrams) > c.relativeMass * fine.massGrams)
        {
            return Decision::refine;
        }
        for (const auto &[name, limit] : c.responseLimits)
        {
            const auto work = fine.responses.at(name);
            const double reference = std::max(std::abs(work), 1e-10 * limit);
            if (std::abs(work - coarse.responses.at(name)) > c.relativeResponse * reference)
            {
                return Decision::refine;
            }
        }
    }
    return exceeded ? Decision::infeasible : Decision::eligible;
}
} // namespace gso
