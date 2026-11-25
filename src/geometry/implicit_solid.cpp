#include "gso/geometry/implicit_solid.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gso
{
namespace
{
void validate(const Box &box)
{
    if (!box.low.allFinite() || !box.high.allFinite() || (box.low.array() >= box.high.array()).any())
    {
        throw std::invalid_argument("Box bounds must be finite and strictly ordered");
    }
}
bool touches(const Box &box, const Point &a, const Point &b)
{
    double enter = 0.0, leave = 1.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        const double delta = b[axis] - a[axis];
        if (delta == 0.0)
        {
            if (a[axis] < box.low[axis] || a[axis] > box.high[axis])
            {
                return false;
            }
        }
        else
        {
            const double first = (box.low[axis] - a[axis]) / delta;
            const double second = (box.high[axis] - a[axis]) / delta;
            enter = std::max(enter, std::min(first, second));
            leave = std::min(leave, std::max(first, second));
            if (enter > leave)
            {
                return false;
            }
        }
    }
    return true;
}
} // namespace
double wendland(double r)
{
    if (r >= 1.0)
    {
        return 0.0;
    }
    const double a = 1.0 - r;
    return a * a * a * a * (1.0 + 4.0 * r);
}
bool Box::contains(const Point &x) const
{
    return (x.array() >= low.array()).all() && (x.array() <= high.array()).all();
}
ImplicitSolid::ImplicitSolid(Box domain, double background, std::vector<Kernel> kernels,
                             std::vector<Box> preserves, std::vector<Box> exclusions)
    : domain_(std::move(domain)), background_(background), kernels_(std::move(kernels)),
      preserves_(std::move(preserves)), exclusions_(std::move(exclusions))
{
    validate(domain_);
    if (!std::isfinite(background_))
    {
        throw std::invalid_argument("Invalid implicit field background");
    }
    for (const auto &k : kernels_)
    {
        if (!k.center.allFinite() || !std::isfinite(k.radius) || k.radius <= 0.0 ||
            !std::isfinite(k.amplitude))
        {
            throw std::invalid_argument("Invalid compact field kernel");
        }
    }
    for (const auto &box : preserves_)
    {
        validate(box);
    }
    for (const auto &box : exclusions_)
    {
        validate(box);
    }
}
bool ImplicitSolid::contains(const Point &x) const
{
    if (!x.allFinite() || !domain_.contains(x))
    {
        return false;
    }
    for (const auto &box : exclusions_)
    {
        if (box.contains(x))
        {
            return false;
        }
    }
    for (const auto &box : preserves_)
    {
        if (box.contains(x))
        {
            return true;
        }
    }
    double value = background_;
    for (const auto &k : kernels_)
    {
        value += k.amplitude * wendland((x - k.center).norm() / k.radius);
    }
    return value > 0.0;
}
Occupancy ImplicitSolid::classify(const Box &cell) const
{
    auto overlaps = [&](const Box &other) {
        return (other.high.array() >= cell.low.array()).all() &&
               (other.low.array() <= cell.high.array()).all();
    };
    if (!overlaps(domain_))
    {
        return Occupancy::empty;
    }
    bool clipped = !domain_.contains(cell.low) || !domain_.contains(cell.high);
    for (const auto &box : exclusions_)
    {
        if (box.contains(cell.low) && box.contains(cell.high))
        {
            return Occupancy::empty;
        }
        clipped = clipped || overlaps(box);
    }
    bool partialPreserve = false;
    for (const auto &box : preserves_)
    {
        if (box.contains(cell.low) && box.contains(cell.high))
        {
            return clipped ? Occupancy::cut : Occupancy::full;
        }
        partialPreserve = partialPreserve || overlaps(box);
    }
    double lo = background_, hi = background_, scale = std::abs(background_);
    for (const auto &k : kernels_)
    {
        const Point nearest = k.center.cwiseMax(cell.low).cwiseMin(cell.high);
        const Point far = (k.center - cell.low).cwiseAbs().cwiseMax((k.center - cell.high).cwiseAbs());
        const double u = k.amplitude * wendland((nearest - k.center).norm() / k.radius);
        const double v = k.amplitude * wendland(far.norm() / k.radius);
        lo += std::min(u, v);
        hi += std::max(u, v);
        scale += std::abs(k.amplitude);
    }
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (lo > tolerance && !clipped)
    {
        return Occupancy::full;
    }
    if (hi < -tolerance && !partialPreserve)
    {
        return Occupancy::empty;
    }
    return Occupancy::cut;
}
bool ImplicitSolid::segmentDefinitelySolid(const Point &a, const Point &b) const
{
    for (const auto &box : preserves_)
    {
        if (box.contains(a) && box.contains(b))
        {
            return true;
        }
    }
    const Point edge = b - a;
    const double length2 = edge.squaredNorm();
    if (length2 == 0.0)
    {
        return contains(a);
    }
    double lower = background_, magnitude = std::abs(background_);
    for (const auto &k : kernels_)
    {
        const double t = std::clamp((k.center - a).dot(edge) / length2, 0.0, 1.0);
        const double distance = k.amplitude < 0.0 ? (a + t * edge - k.center).norm()
                                                  : std::max((a - k.center).norm(), (b - k.center).norm());
        lower += k.amplitude * wendland(distance / k.radius);
        magnitude += std::abs(k.amplitude);
    }
    return lower > 64.0 * std::numeric_limits<double>::epsilon() * magnitude;
}
bool ImplicitSolid::visible(const Point &a, const Point &b) const
{
    if (!contains(a) || !contains(b))
    {
        return false;
    }
    for (const auto &box : exclusions_)
    {
        if (touches(box, a, b))
        {
            return false;
        }
    }
    constexpr unsigned maxDepth = 12;
    struct Segment
    {
        Point first, last;
        unsigned depth;
    };
    std::array<Segment, maxDepth + 1> pending;
    pending[0] = {a, b, 0};
    std::size_t count = 1;
    while (count > 0)
    {
        const Segment s = pending[--count];
        if (segmentDefinitelySolid(s.first, s.last))
        {
            continue;
        }
        const Point middle = (s.first + s.last) * .5;
        if (s.depth == maxDepth || !contains(middle))
        {
            return false;
        }
        pending[count++] = {s.first, middle, s.depth + 1};
        pending[count++] = {middle, s.last, s.depth + 1};
    }
    return true;
}
} // namespace gso
