#pragma once
#include "gso/geometry/implicit_solid.hpp"
#include <array>
#include <cstddef>

namespace gso
{
struct VolumePoint
{
    Point location;
    double volume;
};
struct IntegrationSettings
{
    std::array<unsigned, 3> divisions;
    unsigned boundaryDepth;
    std::size_t maxVisits;
};
struct Quadrature
{
    std::vector<VolumePoint> points;
    double unresolvedVolume = 0.0;
    double interiorVolume = 0.0;
    std::size_t visits = 0;
};
Quadrature integrate(const ImplicitSolid &solid, const IntegrationSettings &settings);
} // namespace gso
