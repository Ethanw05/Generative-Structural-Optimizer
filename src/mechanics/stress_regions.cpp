#include "gso/mechanics/stress_regions.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gso
{
namespace
{
class Plateaus
{
    std::vector<std::size_t> parent_, size_;

public:
    explicit Plateaus(std::size_t count) : parent_(count), size_(count, 1)
    {
        std::iota(parent_.begin(), parent_.end(), std::size_t{0});
    }
    std::size_t root(std::size_t i)
    {
        while (parent_[i] != i)
        {
            parent_[i] = parent_[parent_[i]];
            i = parent_[i];
        }
        return i;
    }
    void join(std::size_t a, std::size_t b)
    {
        a = root(a);
        b = root(b);
        if (a == b) { return; }
        if (size_[b] > size_[a]) { std::swap(a, b); }
        parent_[b] = a;
        size_[a] += size_[b];
    }
};
}
PeakSummary stressMaxima(const std::vector<double> &values, const std::vector<double> &volume,
                         const std::vector<StressEdge> &edges)
{
    const auto count = values.size();
    if (count == 0 || volume.size() != count) { throw std::invalid_argument("Empty stress field"); }
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!std::isfinite(values[i]) || values[i] < 0. || !std::isfinite(volume[i]) || volume[i] <= 0.)
        {
            throw std::invalid_argument("Invalid stress or material volume");
        }
    }
    const double scale = *std::max_element(values.begin(), values.end());
    const double tolerance = std::pow(std::numeric_limits<double>::epsilon(), .75) * scale;
    Plateaus sets(count);
    for (const auto &[a, b] : edges)
    {
        if (a >= count || b >= count || a == b) { throw std::invalid_argument("Invalid stress adjacency"); }
        if (std::abs(values[a] - values[b]) <= tolerance) { sets.join(a, b); }
    }
    std::vector<std::size_t> group(count), peak(count), order;
    std::iota(peak.begin(), peak.end(), std::size_t{0});
    std::vector<double> weight(count, 0.);
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto id = sets.root(i);
        group[i] = id;
        if (weight[id] == 0.) { order.push_back(id); }
        weight[id] += volume[i];
        if (values[i] > values[peak[id]]) { peak[id] = i; }
    }
    std::vector<std::vector<std::size_t>> outlets(count);
    for (const auto &[a, b] : edges)
    {
        auto low = group[a], high = group[b];
        if (values[peak[low]] > values[peak[high]]) { std::swap(low, high); }
        if (values[peak[low]] < values[peak[high]]) { outlets[low].push_back(high); }
    }
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return values[peak[a]] < values[peak[b]]; });
    PeakSummary result;
    double square = 0.;
    for (const auto id : order)
    {
        auto &next = outlets[id];
        if (next.empty())
        {
            const double value = values[peak[id]];
            result.maxima.push_back({peak[id], value, weight[id]});
            if (scale > 0.) { square += weight[id] * (value / scale) * (value / scale); }
            continue;
        }
        std::sort(next.begin(), next.end());
        next.erase(std::unique(next.begin(), next.end()), next.end());
        const auto highest = *std::max_element(next.begin(), next.end(), [&](auto a, auto b) {
            return values[peak[a]] < values[peak[b]];
        });
        const double threshold = values[peak[highest]] - tolerance;
        std::erase_if(next, [&](auto j) { return values[peak[j]] < threshold; });
        const double share = weight[id] / static_cast<double>(next.size());
        for (const auto j : next) { weight[j] += share; }
    }
    result.regionalRmsMPa = scale * std::sqrt(square / std::accumulate(volume.begin(),volume.end(),0.));
    return result;
}
}
