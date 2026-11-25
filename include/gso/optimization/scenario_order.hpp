#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gso
{
// Indices refer to the study's stable, sorted list of scenario names.
std::vector<std::size_t> scenarioOrder(std::size_t count, std::uint64_t seed, std::uint64_t sweep);
}
