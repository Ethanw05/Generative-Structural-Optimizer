#pragma once
#include "gso/geometry/quadrature.hpp"
#include <utility>

namespace gso
{
using StressEdge = std::pair<std::size_t, std::size_t>;

// Two quadrature samples share material across a positive-area face.
std::vector<StressEdge> stressAdjacency(const Quadrature &quadrature, const ImplicitSolid &solid);

struct LocalMaximum
{
    std::size_t sample;
    double peakMPa;
    double basinVolumeMm3;
};

struct PeakSummary
{
    std::vector<LocalMaximum> maxima;
    double regionalRmsMPa = 0.;
};

PeakSummary stressMaxima(const std::vector<double> &values, const std::vector<double> &volume,
                         const std::vector<StressEdge> &edges);
}
