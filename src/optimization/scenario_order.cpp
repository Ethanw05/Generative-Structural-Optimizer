#include "gso/optimization/scenario_order.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace gso
{
std::vector<std::size_t> scenarioOrder(std::size_t count, std::uint64_t seed, std::uint64_t sweep)
{
    if (count == 0) { throw std::invalid_argument("A sweep requires a scenario"); }
    std::seed_seq words{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32),
                        static_cast<std::uint32_t>(sweep), static_cast<std::uint32_t>(sweep >> 32)};
    std::mt19937_64 engine(words);
    std::vector<std::size_t> order(count);
    std::iota(order.begin(), order.end(), std::size_t{0});
    for (auto remaining = count; remaining > 1; --remaining)
    {
        const auto bound = static_cast<std::uint64_t>(remaining);
        // Rejection sampling avoids bias and library-specific shuffle sequences.
        const auto threshold = (std::uint64_t{0} - bound) % bound;
        auto value = engine();
        while (value < threshold) { value = engine(); }
        std::swap(order[remaining - 1], order[static_cast<std::size_t>(value % bound)]);
    }
    return order;
}
}
