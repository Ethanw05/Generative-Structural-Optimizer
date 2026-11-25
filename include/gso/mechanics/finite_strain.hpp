#pragma once
#include <Eigen/Core>

namespace gso
{
struct SolidLaw
{
    double shearMPa;
    double bulkMPa;
    double yieldMPa;  // Zero selects reversible logarithmic elasticity.
    double hardeningMPa;
};

struct PlasticState
{
    Eigen::Matrix3d inversePlasticMetric = Eigen::Matrix3d::Identity();
    double equivalentStrain = 0.;
};

struct SolidResponse
{
    Eigen::Matrix3d firstPiola;
    Eigen::Matrix<double, 9, 9, Eigen::RowMajor> tangent;
    PlasticState next;
    double vonMisesMPa;
    double elasticEnergyDensity;
};

// Total Lagrangian stress and dP/dF. The supplied state is the last committed
// state. A Newton trial never mutates it. Tensor indices in the tangent are row-major.
SolidResponse finiteStrain(const SolidLaw&, const Eigen::Matrix3d& deformation,
                           const PlasticState& committed, bool needTangent);
}
