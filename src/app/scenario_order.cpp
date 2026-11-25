#include "scenario_order.hpp"
#include "gso/optimization/scenario_order.hpp"
#include <charconv>
#include <ostream>
#include <stdexcept>

namespace
{
std::uint64_t integer(std::string_view text)
{
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) { throw std::invalid_argument("Expected an unsigned integer"); }
    return value;
}
}
namespace gso
{
void writeScenarioOrder(std::string_view count, std::string_view seed, std::string_view sweep, std::ostream& out)
{
    for (const auto index : scenarioOrder(integer(count), integer(seed), integer(sweep))) { out << index << '\n'; }
}
}
