#pragma once
#include "gso/mechanics/assembly.hpp"
namespace gso
{
struct AttachmentPoint
{
    Point position;
    double weight;
};
struct Wrench
{
    Point originMm, forceN, momentNmm;
};
// Work-equivalent forces at selected material points, with no artificial rotational restraint.
std::vector<Force> distribute(const Wrench &load, const std::vector<AttachmentPoint> &attachment);
} // namespace gso
