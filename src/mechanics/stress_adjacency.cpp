#include "gso/mechanics/stress_regions.hpp"
#include <map>
#include <stdexcept>

namespace gso
{
std::vector<StressEdge> stressAdjacency(const Quadrature &q, const ImplicitSolid &solid)
{
    using Key = std::pair<unsigned, SampleIndex>;
    std::map<Key, std::size_t> lookup;
    for (std::size_t i = 0; i < q.points.size(); ++i)
    {
        const auto &sample = q.points[i];
        if (!lookup.emplace(Key{sample.level, sample.index}, i).second)
        {
            throw std::invalid_argument("Duplicate integration sample address");
        }
    }
    std::vector<StressEdge> edges;
    for (std::size_t i = 0; i < q.points.size(); ++i)
    {
        const auto &sample = q.points[i];
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            for (int sign : {-1, 1})
            {
                auto neighbor = sample.index;
                neighbor[axis] += sign;
                if (neighbor[axis] < 0) { continue; }
                for (unsigned level = sample.level; level > 0; --level)
                {
                    const auto found = lookup.find({level, neighbor});
                    if (found != lookup.end())
                    {
                        // Equal levels encounter the face twice. A coarse/fine face
                        // is discovered only by its fine side.
                        const auto j = found->second;
                        if ((level < sample.level || i < j) && solid.visible(sample.location, q.points[j].location))
                        {
                            edges.emplace_back(i, j);
                        }
                        break;
                    }
                    for (auto &index : neighbor) { index /= 2; }
                }
            }
        }
    }
    return edges;
}
}
