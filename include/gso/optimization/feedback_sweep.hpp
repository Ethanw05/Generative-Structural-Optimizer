#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace gso
{
using GeometryDigest = std::array<unsigned char, 32>;
// The callback solves the named case on the supplied parent and returns its evaluated descendant.
// Returning the parent retains the geometry for this step. Throwing aborts the sweep.
using CaseUpdate = std::function<GeometryDigest(std::size_t, std::size_t, const GeometryDigest&)>;
GeometryDigest feedbackSweep(std::size_t cases, std::uint64_t seed, std::uint64_t sweep,
                             GeometryDigest current, const CaseUpdate& update);
}
