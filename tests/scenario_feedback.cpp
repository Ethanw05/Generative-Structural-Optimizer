#include "gso/optimization/feedback_sweep.hpp"
#include "gso/optimization/scenario_order.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

int main()
{
    constexpr std::size_t count = 8;
    constexpr std::uint64_t seed = 0x100000001ULL;
    const auto order = gso::scenarioOrder(count, seed, 1);
    auto sorted = order;
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::size_t> expected(count);
    std::iota(expected.begin(), expected.end(), std::size_t{0});
    check(sorted == expected, "Every case must run exactly once in a sweep");
    check(order == gso::scenarioOrder(count, seed, 1), "The seed must replay a sweep");

    gso::GeometryDigest geometry{};
    const auto initial = geometry;
    const auto final = gso::feedbackSweep(count, seed, 1, initial,
        [&](std::size_t index, std::size_t step, const gso::GeometryDigest& parent) {
            check(index == order.at(step - 1), "Mechanical callback received a different scenario");
            check(geometry == parent, "Next case did not receive its predecessor's geometry");
            geometry.at(index) = static_cast<unsigned char>(step);
            return geometry;
        });
    check(final == geometry && final != initial, "The final descendant was lost");
    std::size_t visited = 0;
    try
    {
        gso::feedbackSweep(count, seed, 2, final,
            [&](std::size_t, std::size_t, const gso::GeometryDigest& parent) {
                if (++visited == 3) { throw std::runtime_error("fixture"); }
                return parent;
            });
    }
    catch (const std::runtime_error&) {}
    check(visited == 3, "Sweep continued after the mechanical analysis failed");
}
