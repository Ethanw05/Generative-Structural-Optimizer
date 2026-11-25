#include "front_volume.hpp"
#include "level_set_curvature.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#define GSO_EXPORT __declspec(dllexport)
#else
#define GSO_EXPORT __attribute__((visibility("default")))
#endif

namespace
{
struct ActiveSample
{
    std::size_t index;
    float increment;
    unsigned char edges;
};

float upwind(float middle, float previous, float next, float sign)
{
    return std::max({sign * (middle - previous), sign * (middle - next), 0.f});
}

void transport(std::size_t nx, std::size_t ny, std::size_t nz, const float* field,
               const float* speed, const float* retained, const float* restriction,
               float spacing, float diffusionMm2, float moveBound, float* result)
{
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (nx == 0 || ny == 0 || nz == 0 || ny > limit / nz || nx > limit / (ny * nz) ||
        !std::isfinite(diffusionMm2) || diffusionMm2 < 0.f || !std::isfinite(moveBound) || moveBound < 0.f ||
        (diffusionMm2 > 0.f && moveBound == 0.f) ||
        nz > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / ny ||
        !std::isfinite(spacing) || spacing <= 0.f || !field || !speed || !retained || !restriction || !result)
    {
        throw std::invalid_argument("Invalid front data");
    }
    const auto slab = ny * nz;
    const auto count = nx * slab;
    float maximum = 0.f;
    std::vector<ActiveSample> samples;
    for (std::size_t id = 0; id < count; ++id)
    {
        if (!std::isfinite(field[id]) || !std::isfinite(speed[id]) ||
            !std::isfinite(retained[id]) || !std::isfinite(restriction[id]))
        {
            throw std::invalid_argument("Nonfinite front data");
        }
        result[id] = std::max(std::min(field[id], retained[id]), restriction[id]);
        const bool inCurvatureBand = diffusionMm2 > 0.f && std::abs(field[id]) <= moveBound + 2.f * spacing;
        if ((!inCurvatureBand && speed[id] == 0.f) || restriction[id] >= retained[id]) { continue; }
        const auto y = id / nz % ny;
        const auto z = id % nz;
        const unsigned flags = (id < slab) | ((id >= count - slab) << 1) |
                               ((y == 0) << 2) | ((y + 1 == ny) << 3) |
                               ((z == 0) << 4) | ((z + 1 == nz) << 5);
        samples.push_back({id, speed[id], static_cast<unsigned char>(flags)});
        maximum = std::max(maximum, std::abs(speed[id]));
    }
    if (samples.empty()) { return; }
    // Explicit combined CFL bound for normal transport and tangential diffusion in 3D.
    const double inverseSpacing = 1. / spacing;
    const double diffusion = diffusionMm2 * inverseSpacing * inverseSpacing;
    const double needed = std::ceil(3. * maximum * inverseSpacing + 6. * diffusion);
    if (needed > std::numeric_limits<int>::max()) { throw std::invalid_argument("Unresolved front speed"); }
    const int steps = std::max(1, static_cast<int>(needed));
    for (auto& sample : samples) { sample.increment /= static_cast<float>(steps) * spacing; }
    const float diffusionStep = static_cast<float>(diffusion / steps);
    std::vector<float> update(samples.size());
    for (int step = 0; step < steps; ++step)
    {
        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            const auto [id, speedStep, flags] = samples[i];
            const float middle = result[id];
            const float sign = std::copysign(1.f, speedStep);
            const float sx = upwind(middle, result[(flags & 1) ? id : id - slab],
                                   result[(flags & 2) ? id : id + slab], sign);
            const float sy = upwind(middle, result[(flags & 4) ? id : id - nz],
                                   result[(flags & 8) ? id : id + nz], sign);
            const float sz = upwind(middle, result[(flags & 16) ? id : id - 1],
                                   result[(flags & 32) ? id : id + 1], sign);
            float value = middle - speedStep * std::sqrt(sx*sx + sy*sy + sz*sz);
            if (diffusionStep > 0.f)
            {
                if (!flags)
                {
                    value += diffusionStep * gso::geometry::tangentialLaplacian(
                        result + id, static_cast<std::ptrdiff_t>(slab), static_cast<std::ptrdiff_t>(nz));
                }
                value = std::clamp(value, field[id] - moveBound, field[id] + moveBound);
            }
            update[i] = std::max(std::min(value, retained[id]), restriction[id]);
        }
        for (std::size_t i = 0; i < samples.size(); ++i) { result[samples[i].index] = update[i]; }
    }
}
}

// C ABI for the analysis driver. No host-global pool or worker process is created.
extern "C" GSO_EXPORT int gsoAdvanceFront(std::size_t nx, std::size_t ny, std::size_t nz,
    const float* field, const float* speed, const float* retained, const float* restriction,
    float spacing, float* result) noexcept
{
    try { transport(nx, ny, nz, field, speed, retained, restriction, spacing, 0.f, 0.f, result); }
    catch (const std::invalid_argument&) { return 1; }
    catch (...) { return 2; }
    return 0;
}

extern "C" GSO_EXPORT int gsoAdvanceRegularizedFront(std::size_t nx, std::size_t ny, std::size_t nz,
    const float* field, const float* speed, const float* retained, const float* restriction,
    float spacing, float diffusionMm2, float moveBound, float* result) noexcept
{
    try { transport(nx, ny, nz, field, speed, retained, restriction, spacing, diffusionMm2, moveBound, result); }
    catch (const std::invalid_argument&) { return 1; }
    catch (...) { return 2; }
    return 0;
}

extern "C" GSO_EXPORT int gsoAdvanceVolumeNeutralFront(std::size_t nx, std::size_t ny, std::size_t nz,
    const float* field, const float* speed, const float* retained, const float* restriction,
    float spacing, float diffusionMm2, float moveBound, float* result) noexcept
{
    try
    {
        transport(nx, ny, nz, field, speed, retained, restriction, spacing, diffusionMm2, moveBound, result);
        if (diffusionMm2 > 0.f)
        {
            const auto count = nx * ny * nz;  // Dimensions validated by transport.
            std::vector<float> mechanical(count);
            transport(nx, ny, nz, field, speed, retained, restriction, spacing, 0.f, 0.f, mechanical.data());
            gso::geometry::projectTransportVolume(count, field, mechanical.data(), retained, restriction,
                                                 spacing, moveBound, result);
        }
    }
    catch (const std::invalid_argument&) { return 1; }
    catch (...) { return 2; }
    return 0;
}
