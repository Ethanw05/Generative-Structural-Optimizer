#pragma once
#include "gso/mechanics/assembly.hpp"
namespace gso
{
struct SurfacePatch
{
    Point origin, edgeU, edgeV;
};
std::vector<Restraint> collocateRestraints(const std::vector<Point> &sites, const SurfacePatch &patch,
                                           const std::vector<Point> &directions);
std::vector<Force> integrateTraction(const SurfacePatch &patch, const Point &tractionMPa, double maxPanelMm);
} // namespace gso
