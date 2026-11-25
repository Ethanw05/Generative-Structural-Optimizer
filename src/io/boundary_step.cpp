#include "boundary_step.hpp"
#include "gso/optimization/boundary_step.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace gso
{
void writeBoundaryStep(const std::filesystem::path& input, const std::filesystem::path& output, double scale)
{
    std::ifstream stream(input);
    if (!stream) { throw std::runtime_error("Cannot read boundary samples"); }
    std::vector<BoundarySample> samples;
    while (stream >> std::ws && stream.peek() != std::char_traits<char>::eof())
    {
        BoundarySample sample{};
        if (!(stream >> sample.area >> sample.drivingForce >> sample.minNormalStep >> sample.maxNormalStep))
        {
            throw std::runtime_error("Incomplete boundary sample");
        }
        samples.push_back(sample);
    }
    const auto motion = volumeNeutralStep(samples, scale);
    std::ofstream result(output);
    result.exceptions(std::ios::failbit | std::ios::badbit);
    result << std::setprecision(17);
    for (const double value : motion) { result << value << '\n'; }
}
}
