#include "boundary_step.hpp"
#include "gso/optimization/surface_gradient.hpp"
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace gso
{
void writeSurfaceMotion(const std::filesystem::path& input, const std::filesystem::path& output,
                        double moveBound, double availableVolume)
{
    std::ifstream stream(input);
    long long pointCount = 0;
    long long faceCount = 0;
    if (!(stream >> pointCount >> faceCount) || pointCount <= 0 || faceCount <= 0 ||
        pointCount > std::numeric_limits<int>::max() || faceCount > std::numeric_limits<int>::max())
    {
        throw std::invalid_argument("Invalid surface motion input");
    }
    std::vector<Eigen::Vector3d> points;
    for (long long i = 0; i < pointCount; ++i)
    {
        Eigen::Vector3d point;
        if (!(stream >> point.x() >> point.y() >> point.z())) { throw std::invalid_argument("Missing vertex"); }
        points.push_back(point);
    }
    std::vector<DrivenTriangle> faces;
    for (long long i = 0; i < faceCount; ++i)
    {
        DrivenTriangle face{};
        int lock = 0;
        auto& v = face.vertices;
        if (!(stream >> v[0] >> v[1] >> v[2] >> face.descent >> lock) || (lock != 0 && lock != 1))
        {
            throw std::invalid_argument("Invalid surface motion triangle");
        }
        face.locked = lock != 0;
        faces.push_back(face);
    }
    std::vector<std::array<double, 2>> bounds;
    if ((stream >> std::ws).peek() != std::char_traits<char>::eof())
    {
        std::string label;
        if (!(stream >> label) || label != "bounds") { throw std::invalid_argument("Invalid CAD bounds section"); }
        for (long long i = 0; i < pointCount; ++i)
        {
            std::array<double, 2> limit{};
            if (!(stream >> limit[0] >> limit[1])) { throw std::invalid_argument("Missing CAD bound"); }
            bounds.push_back(limit);
        }
    }
    double regularizationMm = 0.;
    if ((stream >> std::ws).peek() != std::char_traits<char>::eof())
    {
        std::string label;
        if (!(stream >> label >> regularizationMm) || label != "regularization_mm")
        {
            throw std::invalid_argument("Invalid surface regularization length");
        }
    }
    if ((stream >> std::ws).peek() != std::char_traits<char>::eof()) { throw std::invalid_argument("Unexpected surface data"); }
    const auto motion = surfaceMotion(points, faces, moveBound, availableVolume, bounds, regularizationMm);
    std::ofstream result(output);
    result.exceptions(std::ios::failbit | std::ios::badbit);
    result << std::setprecision(17) << motion << '\n';
}
}
