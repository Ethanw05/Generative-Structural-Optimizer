#pragma once
#include "gso/mechanics/assembly.hpp"
#include <complex>

namespace gso
{
struct Excitation
{
    double hertz = 0.0;
    double massDampingPerSecond = 0.0;
    double stiffnessDampingSeconds = 0.0;
};
struct Response
{
    std::complex<double> workNmm;
    Eigen::VectorXcd coefficients;
    Eigen::VectorXcd reactions;
    double relativeResidual;
};
Response solve(const System &system, const Eigen::VectorXd &force, Excitation excitation);
} // namespace gso
