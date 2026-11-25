#include "gso/mechanics/finite_strain.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace gso
{
namespace
{
using Tensor = Eigen::Matrix3d;
Tensor deviator(const Tensor& value)
{
    return value - (value.trace() / 3.) * Tensor::Identity();
}

// Divided difference of log at positive eigenvalues, including repeated stretches.
double logSlope(double a, double b)
{
    if (a == b) { return 1. / a; }
    if (std::abs(a-b) < .5*b) { return std::log1p((a-b)/b) / (a-b); }
    return (std::log(a)-std::log(b)) / (a-b);
}
}

SolidResponse finiteStrain(const SolidLaw& law, const Tensor& f, const PlasticState& prior, bool tangent)
{
    if (!(law.shearMPa > 0.) || !(law.bulkMPa > 0.) || !(law.yieldMPa >= 0.) ||
        !(law.hardeningMPa >= 0.) || !std::isfinite(law.shearMPa+law.bulkMPa+law.yieldMPa+law.hardeningMPa) ||
        !(prior.equivalentStrain >= 0.) || !std::isfinite(prior.equivalentStrain) ||
        !prior.inversePlasticMetric.allFinite() || !f.allFinite())
    {
        throw std::invalid_argument("Invalid finite-strain material or state");
    }
    const double j = f.determinant();
    if (!(j > 0.) || !std::isfinite(j)) { throw std::domain_error("Nonpositive material Jacobian"); }
    const Tensor inverseT = f.inverse().transpose();
    const Tensor trialMetric = f * prior.inversePlasticMetric * f.transpose();
    const Eigen::SelfAdjointEigenSolver<Tensor> eig(trialMetric);
    if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() <= 0.)
    {
        throw std::domain_error("Invalid positive-definite elastic trial metric");
    }
    const auto& v = eig.eigenvectors();
    const auto& stretches = eig.eigenvalues();
    const Tensor logStrain = v * (.5*stretches.array().log()).matrix().asDiagonal() * v.transpose();
    const Tensor trialStress = 2.*law.shearMPa * deviator(logStrain);
    const double q = std::sqrt(1.5) * trialStress.norm();
    const double yield = law.yieldMPa + law.hardeningMPa*prior.equivalentStrain;
    const bool plastic = law.yieldMPa > 0. && q > yield;
    const double modulus = 3.*law.shearMPa + law.hardeningMPa;
    const double increment = plastic ? (q-yield)/modulus : 0.;
    const double factor = plastic ? 1.-3.*law.shearMPa*increment/q : 1.;
    const double logJ = std::log(j);
    const Tensor tau = law.bulkMPa*logJ*Tensor::Identity() + factor*trialStress;
    SolidResponse out{};
    out.firstPiola = tau*inverseT;
    out.next = prior;
    out.tangent.setZero();
    out.vonMisesMPa = factor*q/j;
    out.elasticEnergyDensity = .5*law.bulkMPa*logJ*logJ +
                               factor*factor*trialStress.squaredNorm()/(4.*law.shearMPa);
    if (plastic)
    {
        const Eigen::Vector3d lnTrial = .5*stretches.array().log().matrix();
        const Eigen::Vector3d lnElastic = (lnTrial.array()-lnTrial.mean())*factor + lnTrial.mean();
        const Tensor materialAxes = inverseT.transpose()*v;
        out.next.inversePlasticMetric = materialAxes * (2.*lnElastic.array()).exp().matrix().asDiagonal()
                                         * materialAxes.transpose();
        out.next.equivalentStrain += increment;
    }
    if (!tangent) { return out; }
    Tensor divided;
    for (int i=0; i<3; ++i)
        for (int k=0; k<3; ++k) { divided(i,k)=logSlope(stretches(i),stretches(k)); }
    for (int i=0; i<3; ++i)
    {
        for (int k=0; k<3; ++k)
        {
            Tensor df=Tensor::Zero();
            df(i,k)=1.;
            const Tensor halfDb = df*prior.inversePlasticMetric*f.transpose();
            const Tensor db = halfDb + halfDb.transpose();
            const Tensor dlog = v * divided.cwiseProduct(v.transpose()*db*v) * v.transpose();
            const Tensor dTrial = law.shearMPa * deviator(dlog);
            Tensor dTau = law.bulkMPa*inverseT(i,k)*Tensor::Identity() + factor*dTrial;
            if (plastic)
            {
                const double dq = 1.5*trialStress.cwiseProduct(dTrial).sum()/q;
                dTau -= (3.*law.shearMPa/q)*(1./modulus-increment/q)*dq*trialStress;
            }
            const Tensor dp = (dTau-out.firstPiola*df.transpose())*inverseT;
            for (int a=0; a<3; ++a)
                for (int b=0; b<3; ++b) { out.tangent(3*a+b,3*i+k)=dp(a,b); }
        }
    }
    return out;
}
}
