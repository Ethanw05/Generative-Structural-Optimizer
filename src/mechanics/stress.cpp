#include "gso/mechanics/stress.hpp"
#include "gso/mechanics/approximation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace gso
{
namespace
{
template <class Samples> StressMeasure summarize(const std::vector<double> &volume, double p, Samples sample, std::vector<double> *field)
{
    if (!std::isfinite(p) || p < 1.)
    {
        throw std::invalid_argument("Invalid stress aggregation exponent");
    }
    std::vector<double> scratch;
    auto &values = field ? *field : scratch;
    values.resize(volume.size());
    double peak = 0.;
    for (std::size_t i = 0; i < volume.size(); ++i)
    {
        values[i] = sample(i);
        if (!std::isfinite(values[i]))
        {
            throw std::runtime_error("Nonfinite stress");
        }
        peak = std::max(peak, values[i]);
    }
    if (peak == 0.)
    {
        return {0., 0.};
    }
    double sum = 0.;
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        sum += volume[i] * std::pow(values[i] / peak, p);
    }
    return {peak * std::pow(sum / std::accumulate(volume.begin(), volume.end(), 0.), 1. / p), peak};
}
} // namespace
StressRecovery::StressRecovery(const Model &m, const System &system)
{
    const auto samples = integrate(m.solid, m.integration);
    if (samples.points.empty() ||
        samples.points.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 6))
    {
        throw std::runtime_error("Invalid stress recovery quadrature");
    }
    edges_ = stressAdjacency(samples, m.solid);
    const double shear = m.material.modulusMPa / (2. * (1. + m.material.poisson));
    std::vector<Eigen::Triplet<double>> entries;
    for (const auto &q : samples.points)
    {
        const int row = 6 * static_cast<int>(volume_.size());
        volume_.push_back(q.volume);
        locations_.push_back(q.location);
        for (const auto &shape : approximation(system.sites, m.solid, q.location, m.support))
        {
            for (int a = 0; a < 3; ++a)
            {
                const int col = 3 * shape.site + a;
                for (int b = 0; b < 3; ++b)
                {
                    const double diag =
                        2. * shear * std::sqrt(1.5) * shape.gradient[a] * ((a == b ? 1. : 0.) - 1. / 3.);
                    entries.emplace_back(row + b, col, diag);
                    if (a != b)
                    {
                        entries.emplace_back(row + 6 - a - b, col, shear * std::sqrt(3.) * shape.gradient[b]);
                    }
                }
            }
        }
    }
    tensor_.resize(6 * static_cast<int>(volume_.size()), system.stiffness.cols());
    tensor_.setFromTriplets(entries.begin(), entries.end());
}
StressMeasure StressRecovery::measure(const Eigen::VectorXd &u, double p, std::vector<double> *field) const
{
    if (u.size() != tensor_.cols())
    {
        throw std::invalid_argument("Invalid displacement size for stress recovery");
    }
    const Eigen::VectorXd stress = tensor_ * u;
    return summarize(volume_, p, [&](std::size_t i) { return stress.segment<6>(6 * i).norm(); }, field);
}
StressMeasure StressRecovery::measureHarmonic(const Eigen::VectorXcd &u, double p, std::vector<double> *field) const
{
    if (u.size() != tensor_.cols())
    {
        throw std::invalid_argument("Invalid harmonic displacement size");
    }
    const Eigen::VectorXd real = tensor_ * u.real(), imag = tensor_ * u.imag();
    return summarize(volume_, p, [&](std::size_t i) {
        const auto a = real.segment<6>(6 * i), b = imag.segment<6>(6 * i);
        const double aa = a.squaredNorm(), bb = b.squaredNorm();
        return std::sqrt(.5 * (aa + bb + std::hypot(aa - bb, 2. * a.dot(b))));
    }, field);
}
PeakSummary StressRecovery::peaks(const std::vector<double> &field) const
{
    return stressMaxima(field, volume_, edges_);
}
std::vector<double> StressRecovery::regionPeaks(const std::vector<double> &field,
                                              const std::vector<Point> &anchors) const
{
    if (field.size() != locations_.size() || anchors.empty())
    {
        throw std::invalid_argument("Missing material field or reference regions");
    }
    for (const auto &anchor : anchors)
    {
        if (!anchor.allFinite()) { throw std::invalid_argument("Invalid reference region"); }
    }
    std::vector<double> result(anchors.size(), 0.);
    std::vector<double> distance(anchors.size());
    for (std::size_t i = 0; i < field.size(); ++i)
    {
        for (std::size_t j = 0; j < anchors.size(); ++j)
        {
            distance[j] = (anchors[j]-locations_[i]).squaredNorm();
        }
        const double closest = *std::min_element(distance.begin(),distance.end());
        const double tie = 32.*std::numeric_limits<double>::epsilon()*closest;
        for (std::size_t j = 0; j < anchors.size(); ++j)
        {
            if (distance[j] <= closest+tie) { result[j] = std::max(result[j], field[i]); }
        }
    }
    return result;
}
} // namespace gso
