#include "gso/optimization/boundary_step.hpp"
#include <array>
#include <cmath>
#include <stdexcept>

int main()
{
    const std::array<gso::BoundarySample,4> samples{{{1.,9.,-1.,1.},{2.,0.,-1.,1.},
                                                    {1.,2.,0.,0.},{1.,0.,0.,1.}}};
    const auto move = gso::volumeNeutralStep(samples, 1.);
    if (std::abs(move[0]-1.) > 1.e-12 || std::abs(move[1]+.5) > 1.e-12 ||
        move[2] != 0. || move[3] != 0.) { return 1; }
    try { (void)gso::volumeNeutralStep(samples,-1.); }
    catch (const std::invalid_argument&) { return 0; }
    return 1;
}
