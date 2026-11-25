#include "gso/optimization/feedback_sweep.hpp"
#include "gso/optimization/scenario_order.hpp"
#include <stdexcept>

namespace gso
{
GeometryDigest feedbackSweep(std::size_t cases, std::uint64_t seed, std::uint64_t sweep,
                             GeometryDigest current, const CaseUpdate& update)
{
    if (!update) { throw std::invalid_argument("A mechanical update is required"); }
    const auto sequence = scenarioOrder(cases, seed, sweep);
    for (std::size_t step = 0; step < sequence.size(); ++step)
    {
        current = update(sequence[step], step + 1, current);
    }
    return current;
}
}
