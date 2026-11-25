#pragma once
#include "gso/geometry/implicit_solid.hpp"
#include <vector>

namespace gso
{
struct Shape
{
    int site;
    double value;
    Point gradient;
};
// First-order moving least squares with differentiated moment equations.
std::vector<Shape> approximation(const std::vector<Point> &sites, const ImplicitSolid &solid,
                                 const Point &location, double support);
} // namespace gso
