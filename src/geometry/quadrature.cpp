#include "gso/geometry/quadrature.hpp"
#include <cmath>
#include <stdexcept>

namespace gso
{
namespace
{
constexpr unsigned maxCutDepth = 12;
void visit(const ImplicitSolid &solid, const Box &root, unsigned remaining, std::size_t budget,
           const SampleIndex &rootIndex, Quadrature &out)
{
    struct Cell
    {
        Box box;
        unsigned depth;
        SampleIndex index;
    };
    // Depth-first subdivision needs at most seven deferred siblings per level.
    std::array<Cell, 1 + 7 * maxCutDepth> pending;
    pending[0] = {root, remaining, rootIndex};
    std::size_t count = 1;
    while (count > 0)
    {
        const auto [box, depth, index] = pending[--count];
        if (++out.visits > budget)
        {
            throw std::runtime_error("Integration subdivision budget exhausted");
        }
        const auto state = solid.classify(box);
        if (state == Occupancy::empty)
        {
            continue;
        }
        const Point mid = (box.low + box.high) * 0.5;
        if (state == Occupancy::cut && depth > 0)
        {
            for (unsigned mask = 0; mask < 8; ++mask)
            {
                Box child = box;
                SampleIndex childIndex{};
                for (int a = 0; a < 3; ++a)
                {
                    childIndex[a] = 2 * index[a] + ((mask >> a) & 1U);
                    if ((mask & (1U << a)) != 0)
                    {
                        child.low[a] = mid[a];
                    }
                    else
                    {
                        child.high[a] = mid[a];
                    }
                }
                pending[count++] = {child, depth - 1, childIndex};
            }
            continue;
        }
        const Point side = box.high - box.low;
        const double volume = side.prod();
        if (state == Occupancy::cut)
        {
            out.unresolvedVolume += volume;
        }
        else
        {
            out.interiorVolume += volume;
        }
        for (unsigned mask = 0; mask < 8; ++mask)
        {
            Point p = mid;
            for (int a = 0; a < 3; ++a)
            {
                p[a] += ((mask & (1U << a)) ? 1.0 : -1.0) * side[a] / std::sqrt(12.0);
            }
            if (state == Occupancy::full || solid.contains(p))
            {
                SampleIndex address{};
                for (int a = 0; a < 3; ++a) { address[a] = 2 * index[a] + ((mask >> a) & 1U); }
                out.points.push_back({p, volume / 8.0, address, remaining - depth + 1});
            }
        }
    }
}
} // namespace
Quadrature integrate(const ImplicitSolid &solid, const IntegrationSettings &s)
{
    if (s.boundaryDepth > maxCutDepth || s.maxVisits == 0)
    {
        throw std::invalid_argument("Invalid integration budget");
    }
    const Box &domain = solid.bounds();
    Point step;
    std::size_t count = 1;
    for (int a = 0; a < 3; ++a)
    {
        if (s.divisions[a] == 0 || count > s.maxVisits / s.divisions[a])
        {
            throw std::invalid_argument("Invalid integration divisions or budget");
        }
        count *= s.divisions[a];
        step[a] = (domain.high[a] - domain.low[a]) / s.divisions[a];
    }
    Quadrature out;
    for (unsigned i = 0; i < s.divisions[0]; ++i)
    {
        for (unsigned j = 0; j < s.divisions[1]; ++j)
        {
            for (unsigned k = 0; k < s.divisions[2]; ++k)
            {
                const Point index(i, j, k);
                Box cell{domain.low + step.cwiseProduct(index),
                         domain.low + step.cwiseProduct(index + Point::Ones())};
                for (int a = 0; a < 3; ++a)
                {
                    if (index[a] + 1.0 == s.divisions[a])
                    {
                        cell.high[a] = domain.high[a];
                    }
                }
                visit(solid, cell, s.boundaryDepth, s.maxVisits, {i, j, k}, out);
            }
        }
    }
    return out;
}
} // namespace gso
