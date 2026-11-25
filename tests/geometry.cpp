#include "gso/geometry/quadrature.hpp"
#include "test_support.hpp"
#include <numbers>

int main()
{
    using namespace gso;
    const Box box{Point::Zero(), Point::Ones()};
    const Kernel hole{{.231, .51, .5}, .2, -1.};
    ImplicitSolid solid(box, .1875, {hole});
    check(!solid.contains(hole.center), "Hole center has material");
    check(solid.contains({.5, .51, .5}), "Solid interior removed");
    check(!solid.visible({0, .51, .5}, {1, .51, .5}), "Support crosses an off-center cavity");
    check(solid.visible({0, .8, .5}, {1, .8, .5}), "Interior support lost");
    ImplicitSolid protectedSolid(box, -1., {}, {{{0, 0, 0}, {.5, .5, .5}}}, {{{.1, .1, .1}, {.2, .2, .2}}});
    check(protectedSolid.contains({.3, .3, .3}) && !protectedSolid.contains({.15, .15, .15}),
          "Region priority");
    const double exact = 1. - 4. * std::numbers::pi * .001 / 3.;
    double previous = 1.;
    for (unsigned depth : {2U, 3U, 4U})
    {
        const auto q = integrate(solid, {{2, 2, 2}, depth, 100000});
        double volume = 0.;
        for (const auto &p : q.points)
        {
            volume += p.volume;
        }
        check(q.interiorVolume <= exact + 1e-12 && exact <= q.interiorVolume + q.unresolvedVolume + 1e-12,
              "Sphere volume enclosure");
        check(std::abs(volume - exact) <= q.unresolvedVolume + 1e-12,
              "Volume outside integration uncertainty");
        check(q.unresolvedVolume < previous, "Unresolved boundary failed to shrink");
        previous = q.unresolvedVolume;
    }
    rejects([&] { integrate(solid, {{2, 2, 2}, 3, 8}); }, "Budget exhaustion was silent");
}
