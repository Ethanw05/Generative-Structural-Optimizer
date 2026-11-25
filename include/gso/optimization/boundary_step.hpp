#pragma once

#include <span>
#include <vector>

namespace gso
{
struct BoundarySample
{
    double area;
    double drivingForce;
    double minNormalStep;
    double maxNormalStep;
};

// Project the unconstrained normal motion onto sum(area * step) = 0.
std::vector<double> volumeNeutralStep(std::span<const BoundarySample> surface, double scale);
}
