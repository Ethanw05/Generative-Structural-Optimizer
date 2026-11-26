#include "gso/mechanics/transient.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

#if defined(_WIN32)
#define GSO_MECHANICS_EXPORT __declspec(dllexport)
#else
#define GSO_MECHANICS_EXPORT __attribute__((visibility("default")))
#endif

// Dense Galerkin matrices are supplied in row-major order. All time integration
// and unilateral contact use the same operator as a native full-order study.
namespace
{
int recordHistory(int modes, int records, const double *times,
    const double *stiffness, const double *mass, const double *velocity, int loads,
    const double *spatial, const double *series, int contacts, const double *normals,
    const double *gaps, const double *springs, double massDamping, double stiffnessDamping,
    double *states, double *response, double *accelerations) noexcept
{
    if (modes<1 || records<2 || contacts<0 || loads<0 || !times ||
        !stiffness || !mass || !velocity || !states || !response ||
        (loads && (!spatial || !series)) || (contacts && (!normals || !gaps || !springs))) { return 1; }
    try
    {
        using Matrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
        using Vector=Eigen::VectorXd;
        gso::System system;
        system.stiffness=Eigen::Map<const Matrix>(stiffness,modes,modes).sparseView();
        system.mass=Eigen::Map<const Matrix>(mass,modes,modes).sparseView();
        system.constraints.resize(0,modes);
        if (times[0]!=0.) { return 1; }
        for (int j=1; j<records; ++j)
        {
            if (!std::isfinite(times[j]) || times[j]<=times[j-1]) { return 1; }
        }
        // Every requested instant is an integration knot, including contact-only records.
        gso::TransientCase scenario{times[records-1],times[records-1],
            Eigen::Map<const Vector>(velocity,modes),{},{},massDamping,stiffnessDamping};
        for (int i=0; i<std::max(1,loads); ++i)
        {
            gso::TimeLoad load{Vector::Zero(modes),{}};
            if (loads) { load.spatialForce=Eigen::Map<const Vector>(spatial+std::size_t(i)*modes,modes); }
            for (int j=0; j<records; ++j)
            {
                load.history.push_back({times[j],loads ? series[std::size_t(i)*records+j] : 0.});
            }
            scenario.loads.push_back(std::move(load));
        }
        for (int i=0; i<contacts; ++i)
        {
            const Vector direction=Eigen::Map<const Vector>(normals+std::size_t(i)*modes,modes);
            scenario.contacts.push_back({direction.sparseView(),gaps[i],springs[i]});
        }
        std::fill_n(states,modes,0.);
        int next=1;
        int accelerationRow=0;
        std::function<void(double, const Vector&)> recordAcceleration;
        if (accelerations)
        {
            recordAcceleration=[&](double time, const Vector &a) {
                if (accelerationRow>=records || time!=times[accelerationRow]) {
                    throw std::runtime_error("Inconsistent acceleration recording time");
                }
                Eigen::Map<Vector> row(accelerations+std::size_t(accelerationRow)*modes,modes);
                row=a;
                ++accelerationRow;
            };
        }
        const auto run=gso::integrateTimedHistory(system,scenario,{},[&](double time,const Vector &u) {
            if (next>=records || time!=times[next])
            {
                throw std::runtime_error("Inconsistent transient recording time");
            }
            Eigen::Map<Vector> sample(states+std::size_t(next)*modes,modes);
            sample=u;
            ++next;
        },recordAcceleration);
        if (next!=records || (accelerations && accelerationRow!=records)) { return 2; }
        response[0]=run.peakStrainEnergyNmm;
        response[1]=run.peakContactForceN;
        response[2]=run.maxResidual;
        return 0;
    }
    catch (...) { return 3; }
}

}

extern "C" GSO_MECHANICS_EXPORT int gsoResponseHistoryAt(int modes, int records, const double *times,
    const double *stiffness, const double *mass, const double *velocity, int loads,
    const double *spatial, const double *series, int contacts, const double *normals,
    const double *gaps, const double *springs, double massDamping, double stiffnessDamping,
    double *states, double *response) noexcept
{
    return recordHistory(modes,records,times,stiffness,mass,velocity,loads,spatial,series,contacts,normals,
                         gaps,springs,massDamping,stiffnessDamping,states,response,nullptr);
}

extern "C" GSO_MECHANICS_EXPORT int gsoResponseKinematicsAt(int modes, int records, const double *times,
    const double *stiffness, const double *mass, const double *velocity, int loads,
    const double *spatial, const double *series, int contacts, const double *normals,
    const double *gaps, const double *springs, double massDamping, double stiffnessDamping,
    double *states, double *response, double *accelerations) noexcept
{
    if (!accelerations) { return 1; }
    return recordHistory(modes,records,times,stiffness,mass,velocity,loads,spatial,series,contacts,normals,
                         gaps,springs,massDamping,stiffnessDamping,states,response,accelerations);
}
