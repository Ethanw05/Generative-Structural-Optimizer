#include "gso/mechanics/stress_regions.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <numeric>

int main()
{
    using namespace gso;
    // Symmetric competing outlets share the same material, including after renumbering.
    const std::vector<StressEdge> path{{0,1},{1,2},{2,3},{3,4}};
    const auto split = stressMaxima({7.,3.,1.,3.,7.}, {1.,1.,2.,1.,1.}, path);
    check(split.maxima.size() == 2, "Lost a local maximum");
    for (const auto &peak : split.maxima) { near(peak.basinVolumeMm3, 3., 1e-12, "Biased watershed divide"); }
    near(split.regionalRmsMPa, 7., 1.e-12, "Equal regional peaks");
    const auto unequal = stressMaxima({100.,1.,5.,5.,2.},{1.,1.,1.,1.,1.},path);
    check(unequal.maxima.size() == 2, "A larger remote stress masked a lower plateau");
    near(unequal.regionalRmsMPa, std::sqrt((2.*10000.+3.*25.)/5.), 1.e-12, "Regional peak aggregation");
    const auto disconnected = stressMaxima({0.,0.,0.},{1.,1.,1.},{{0,1}});
    check(disconnected.maxima.size() == 2, "Disconnected plateaus were joined");

    const Box box{{0.,0.,0.},{2.,1.,1.}};
    const ImplicitSolid block(box,1.);
    const IntegrationSettings settings{{2,1,1},1,256};
    const auto samples = integrate(block,settings);
    const auto adjacency = stressAdjacency(samples,block);
    check(samples.points.size()==16 && adjacency.size()==28,"Incorrect integration lattice adjacency");
    const Box gap{{.96,0.,0.},{1.04,1.,1.}};
    const ImplicitSolid divided(box,1.,{},{},{gap});
    const auto cut = integrate(divided,settings);
    const auto edges = stressAdjacency(cut,divided);
    for (const auto &[a,b] : edges)
    {
        const double x=cut.points[a].location.x()-1., y=cut.points[b].location.x()-1.;
        check(x*y>0.,"Stress neighborhood crossed a physical exclusion");
    }
    const auto parts=stressMaxima(std::vector<double>(cut.points.size(),1.),
                                  std::vector<double>(cut.points.size(),1.),edges);
    check(parts.maxima.size()==2,"Coarse and fine material samples lost face connectivity");
}
