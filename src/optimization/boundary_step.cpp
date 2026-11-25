#include "gso/optimization/boundary_step.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gso
{
std::vector<double> volumeNeutralStep(std::span<const BoundarySample> surface, double scale)
{
    if (surface.empty() || !std::isfinite(scale) || scale <= 0.)
    {
        throw std::invalid_argument("Invalid boundary step");
    }
    double low = std::numeric_limits<double>::infinity();
    double high = -low;
    for (const auto& sample : surface)
    {
        if (!std::isfinite(sample.area) || sample.area <= 0. || !std::isfinite(sample.drivingForce) ||
            !std::isfinite(sample.minNormalStep) || !std::isfinite(sample.maxNormalStep) ||
            sample.minNormalStep > 0. || sample.maxNormalStep < 0.)
        {
            throw std::invalid_argument("Invalid boundary evidence or motion constraints");
        }
        low = std::min(low, scale * sample.drivingForce - sample.maxNormalStep);
        high = std::max(high, scale * sample.drivingForce - sample.minNormalStep);
    }
    auto stepAt = [scale](const BoundarySample& sample, double multiplier)
    {
        return std::clamp(scale * sample.drivingForce - multiplier,
                          sample.minNormalStep, sample.maxNormalStep);
    };
    for (unsigned iteration = 0; iteration < 64; ++iteration)
    {
        const double multiplier = .5 * (low + high);
        double change = 0.;
        for (const auto& sample : surface)
        {
            change += sample.area * stepAt(sample, multiplier);
        }
        if (change > 0.) { low = multiplier; }
        else { high = multiplier; }
    }
    std::vector<double> result;
    result.reserve(surface.size());
    for (const auto& sample : surface)
    {
        result.push_back(stepAt(sample, .5 * (low + high)));
    }
    return result;
}
}
