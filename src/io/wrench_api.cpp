#include "gso/mechanics/wrench.hpp"
#include <cstdint>
#include <limits>

#if defined(_WIN32)
#define GSO_MECHANICS_EXPORT __declspec(dllexport)
#else
#define GSO_MECHANICS_EXPORT __attribute__((visibility("default")))
#endif

extern "C" GSO_MECHANICS_EXPORT int gsoDistributeWrench(std::size_t count, const double *positions,
    const double *weights, const double *origin, const double *force, const double *moment, double *out) noexcept
{
    if (!count || count > std::numeric_limits<std::size_t>::max()/3 || !positions ||
        !weights || !origin || !force || !moment || !out) { return 1; }
    try
    {
        std::vector<gso::AttachmentPoint> points;
        points.reserve(count);
        for (std::size_t i=0; i<count; ++i)
        {
            points.push_back({Eigen::Map<const gso::Point>(positions+3*i), weights[i]});
        }
        const gso::Wrench load{Eigen::Map<const gso::Point>(origin),Eigen::Map<const gso::Point>(force),
                               Eigen::Map<const gso::Point>(moment)};
        const auto forces = gso::distribute(load, points);
        for (std::size_t i=0; i<count; ++i)
        {
            Eigen::Map<gso::Point> target(out+3*i);
            target = forces[i].newtons;
        }
        return 0;
    }
    catch (...) { return 2; }
}
