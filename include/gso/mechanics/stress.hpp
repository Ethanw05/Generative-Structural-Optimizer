#pragma once
#include "gso/mechanics/assembly.hpp"
#include "gso/mechanics/stress_regions.hpp"
#include <complex>
namespace gso
{
struct StressMeasure
{
    double aggregateMPa, peakMPa;
};
// Each six-row block recovers the weighted deviatoric stress tensor at one solid integration point.
class StressRecovery
{
  public:
    StressRecovery(const Model &model, const System &system);
    StressMeasure measure(const Eigen::VectorXd &u, double exponent, std::vector<double> *field = nullptr) const;
    StressMeasure measureHarmonic(const Eigen::VectorXcd &u, double exponent, std::vector<double> *field = nullptr) const;
    PeakSummary peaks(const std::vector<double> &field) const;
    std::vector<double> regionPeaks(const std::vector<double> &field, const std::vector<Point> &anchors) const;
    const Point &location(std::size_t sample) const { return locations_.at(sample); }

  private:
    Sparse tensor_;
    std::vector<Point> locations_;
    std::vector<StressEdge> edges_;
    std::vector<double> volume_;
};
} // namespace gso
