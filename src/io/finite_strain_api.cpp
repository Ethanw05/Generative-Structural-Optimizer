#include "gso/mechanics/finite_strain.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#if defined(_WIN32)
#define GSO_EXPORT __declspec(dllexport)
#else
#define GSO_EXPORT __attribute__((visibility("default")))
#endif

extern "C" GSO_EXPORT int gsoSolidPoints(std::size_t n, const double* properties, const double* deformation,
    const double* committed, double* next, double* piola, double* measures, double* tangent) noexcept
{
    if (n==0 || !properties || !deformation || !committed || !next || !piola || !measures) { return 1; }
    using RowMatrix = Eigen::Matrix<double,3,3,Eigen::RowMajor>;
    const gso::SolidLaw law{properties[0],properties[1],properties[2],properties[3]};
    try
    {
        for (std::size_t i=0; i<n; ++i)
        {
            const gso::PlasticState prior{Eigen::Map<const RowMatrix>(committed+10*i),committed[10*i+9]};
            const auto value=gso::finiteStrain(law,Eigen::Map<const RowMatrix>(deformation+9*i),prior,tangent!=nullptr);
            Eigen::Map<RowMatrix> stress(piola+9*i);
            Eigen::Map<RowMatrix> state(next+10*i);
            stress=value.firstPiola;
            state=value.next.inversePlasticMetric;
            next[10*i+9]=value.next.equivalentStrain;
            measures[2*i]=value.vonMisesMPa;
            measures[2*i+1]=value.elasticEnergyDensity;
            if (tangent) { std::copy_n(value.tangent.data(),81,tangent+81*i); }
        }
        return 0;
    }
    catch (const std::domain_error&) { return 2; }
    catch (...) { return 1; }
}
