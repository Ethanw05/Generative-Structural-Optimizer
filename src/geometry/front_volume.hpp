#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace gso::geometry
{
inline double occupiedFraction(float value, float h)
{
    return std::clamp(.5 - .5 * static_cast<double>(value) / h, 0., 1.);
}

// Project curvature motion onto the material volume requested by mechanics.
// Both the reference volume and every offset respect CAD and the trust region.
inline void projectTransportVolume(std::size_t count, const float* source, const float* mechanical,
                                  const float* keep, const float* outside, float h, float move,
                                  float* curved)
{
    struct Interval { std::size_t id; float lower; float upper; };
    std::vector<Interval> interface;
    auto bounds = [&](std::size_t i, float offset) {
        return std::max(std::min(source[i] + offset, keep[i]), outside[i]);
    };
    double target = 0.;
    for (std::size_t i = 0; i < count; ++i)
    {
        const float lo = bounds(i, -move);
        const float hi = bounds(i, move);
        if (lo >= hi || lo >= h || hi <= -h) { continue; }
        interface.push_back({i, lo, hi});
        target += occupiedFraction(std::clamp(mechanical[i], lo, hi), h);
    }
    if (interface.empty()) { return; }
    auto residual = [&](float offset) {
        double volume = 0.;
        for (const auto& cell : interface)
        {
            const float value = std::clamp(curved[cell.id] - offset, cell.lower, cell.upper);
            volume += occupiedFraction(value, h);
        }
        return volume - target;
    };
    const double initial = residual(0.f);
    if (std::abs(initial) < 1.e-10) { return; }
    float low = initial < 0. ? 0.f : -2.f * move;
    float high = initial < 0. ? 2.f * move : 0.f;
    for (int iteration = 0; iteration < std::numeric_limits<float>::digits + 2; ++iteration)
    {
        const float middle = .5f * (low + high);
        if (residual(middle) < 0.) { low = middle; }
        else { high = middle; }
    }
    const float offset = .5f * (high + low);
    for (std::size_t i = 0; i < count; ++i)
    {
        curved[i] = std::clamp(curved[i] - offset, bounds(i, -move), bounds(i, move));
    }
}
}
