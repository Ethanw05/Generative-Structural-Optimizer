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
               float spacing, float* result)
{
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (nx == 0 || ny == 0 || nz == 0 || ny > limit / nz || nx > limit / (ny * nz) ||
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
        if (speed[id] == 0.f) { continue; }
        const auto y = id / nz % ny;
        const auto z = id % nz;
        const unsigned flags = (id < slab) | ((id >= count - slab) << 1) |
                               ((y == 0) << 2) | ((y + 1 == ny) << 3) |
                               ((z == 0) << 4) | ((z + 1 == nz) << 5);
        samples.push_back({id, speed[id], static_cast<unsigned char>(flags)});
        maximum = std::max(maximum, std::abs(speed[id]));
    }
    if (samples.empty()) { return; }
    const double needed = std::ceil(3. * maximum / spacing);
    if (needed > std::numeric_limits<int>::max()) { throw std::invalid_argument("Unresolved front speed"); }
    const int steps = std::max(1, static_cast<int>(needed));
    for (auto& sample : samples) { sample.increment /= static_cast<float>(steps) * spacing; }
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
            update[i] = std::max(std::min(middle - speedStep * std::sqrt(sx*sx + sy*sy + sz*sz),
                                          retained[id]), restriction[id]);
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
    try { transport(nx, ny, nz, field, speed, retained, restriction, spacing, result); }
    catch (const std::invalid_argument&) { return 1; }
    catch (...) { return 2; }
    return 0;
}
