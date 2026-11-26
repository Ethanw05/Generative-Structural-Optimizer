#pragma once
#include "gso/mechanics/assembly.hpp"
#include <functional>
namespace gso
{
struct LoadKnot
{
    double seconds, multiplier;
};
struct TimeLoad
{
    Eigen::VectorXd spatialForce;
    std::vector<LoadKnot> history;
};
// A row maps generalized displacements into the signed normal gap at a contact sample.
// The direction can also encode relative motion between two bodies.
struct NormalContact
{
    Eigen::SparseVector<double> direction;
    double initialGapMm, stiffnessNPerMm;
};
struct TransientCase
{
    double endSeconds, maxStepSeconds;
    Eigen::VectorXd initialVelocity;
    std::vector<TimeLoad> loads;
    std::vector<NormalContact> contacts;
    double massDampingPerSecond = 0., stiffnessDampingSeconds = 0.;
};
struct TimeResponse
{
    double peakStrainEnergyNmm = 0., peakContactForceN = 0., peakProbeMm = 0., maxResidual = 0.;
    unsigned steps = 0;
    Eigen::VectorXd displacement, velocity;
};
// Probe is the work-conjugate interpolation vector for a unit directional force.
TimeResponse integrateHistory(const System &system, const TransientCase &scenario,
                              const Eigen::VectorXd &probe = {},
                              const std::function<void(const Eigen::VectorXd &)> &observe = {});
TimeResponse integrateTimedHistory(const System &system, const TransientCase &scenario, const Eigen::VectorXd &probe,
                                   const std::function<void(double, const Eigen::VectorXd &)> &observe,
                                   const std::function<void(double, const Eigen::VectorXd &)> &observeAcceleration = {});
} // namespace gso
