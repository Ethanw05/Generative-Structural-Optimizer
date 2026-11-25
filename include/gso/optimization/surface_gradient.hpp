#pragma once
#include <Eigen/Core>
#include <array>
#include <span>
#include <vector>

namespace gso
{
struct DrivenTriangle
{
    std::array<int, 3> vertices;
    double descent;
    bool locked;
};

// P1 surface Sobolev gradient. The move bound is in mm and the available volume in mm^3.
Eigen::VectorXd surfaceMotion(std::span<const Eigen::Vector3d> points,
                              std::span<const DrivenTriangle> faces,
                              double moveBound, double availableVolume,
                              std::span<const std::array<double, 2>> bounds = {},
                              double regularizationMm = 0.);
}
