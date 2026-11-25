#include "global_starts.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gso
{
GlobalStarts::GlobalStarts(std::size_t dimensions)
{
    for (unsigned candidate = 2; primes_.size() < dimensions; ++candidate)
    {
        bool prime = true;
        for (const auto divisor : primes_)
        {
            if (divisor > candidate / divisor) { break; }
            if (candidate % divisor == 0) { prime = false; break; }
        }
        if (prime) { primes_.push_back(candidate); }
        if (candidate == std::numeric_limits<unsigned>::max())
        {
            throw std::invalid_argument("Too many independent design coordinates");
        }
    }
}

std::vector<double> GlobalStarts::at(unsigned index, const std::vector<DesignAxis> &axes) const
{
    if (index == 0 || axes.size() != primes_.size())
    {
        throw std::invalid_argument("Invalid global start");
    }
    std::vector<double> result;
    result.reserve(axes.size());
    for (std::size_t dimension = 0; dimension < axes.size(); ++dimension)
    {
        const auto base = primes_[dimension];
        double fraction = 1. / base;
        double unit = 0.;
        for (auto digits = index; digits != 0; digits /= base)
        {
            unit += (digits % base) * fraction;
            fraction /= base;
        }
        result.push_back(std::lerp(axes[dimension].minimum, axes[dimension].maximum, unit));
    }
    return result;
}
} // namespace gso
