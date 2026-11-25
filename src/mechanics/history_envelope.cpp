#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(_WIN32)
#define GSO_EXPORT __declspec(dllexport)
#else
#define GSO_EXPORT __attribute__((visibility("default")))
#endif

// For each spatial location, recover max_t ||A q(t)|| and its first time index.
// The triangle inequality gives a rigorous upper bound for an entire time block:
// sum_j ||A.col(j)|| max_t |q_j(t)|. All surviving records are evaluated in full.
extern "C" GSO_EXPORT int gsoHistoryEnvelope(
    int sites, int components, int modes, int records, int blockSize, const double *maps,
    const double *states, const double *bounds, const std::int64_t *timeIndex,
    double *maximum, std::int64_t *when) noexcept
{
    if (sites<1 || components<1 || modes<1 || records<1 || blockSize<1 ||
        !maps || !states || !bounds || !timeIndex || !maximum || !when) { return 1; }
    using Matrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
    using Vector=Eigen::VectorXd;
    try
    {
        const Eigen::Map<const Matrix> history(states,records,modes);
        const int blocks=1+(records-1)/blockSize;
        const Eigen::Map<const Matrix> bound(bounds,blocks,modes);
        const double margin=1.+64.*std::numeric_limits<double>::epsilon()*(double(modes)+components);
        Vector norms(modes), sample(components);
        for (int site=0; site<sites; ++site)
        {
            const Eigen::Map<const Matrix> map(maps+std::size_t(site)*components*modes,components,modes);
            norms=map.colwise().norm().transpose();
            double best=0., bestSquared=0.;
            std::int64_t first=0;
            for (int block=0; block<blocks; ++block)
            {
                if (bound.row(block).dot(norms)*margin<best) { continue; }
                const int begin=block*blockSize;
                const int end=begin+std::min(blockSize,records-begin);
                for (int t=begin; t<end; ++t)
                {
                    sample.noalias()=map*history.row(t).transpose();
                    const double value=sample.squaredNorm();
                    if (value>bestSquared || (value==bestSquared && timeIndex[t]<first))
                    {
                        bestSquared=value;
                        first=timeIndex[t];
                    }
                }
                best=std::sqrt(bestSquared);
            }
            maximum[site]=best;
            when[site]=first;
        }
        return 0;
    }
    catch (...) { return 2; }
}
