#include "gso/mechanics/approximation.hpp"
#include "test_support.hpp"
#include <map>

int main()
{
    using namespace gso;
    const ImplicitSolid solid({{-2, -2, -2}, {2, 2, 2}}, 1.);
    std::vector<Point> sites = {{0, 0, 0},     {.8, .2, .1},  {.1, 1., .15}, {-.2, .1, .9},
                                {-.6, .4, .3}, {.2, -.5, .4}, {.6, .7, -.4}};
    const Point x(.19, .23, .17);
    const auto shape = approximation(sites, solid, x, 1.8);
    double sum = 0.;
    Point position = Point::Zero(), gradient = Point::Zero();
    Eigen::Matrix3d identity = Eigen::Matrix3d::Zero();
    for (const auto &s : shape)
    {
        sum += s.value;
        position += s.value * sites[s.site];
        gradient += s.gradient;
        identity += sites[s.site] * s.gradient.transpose();
    }
    near(sum, 1., 2e-13, "Partition of unity");
    check((position - x).norm() < 2e-13 && gradient.norm() < 2e-12 &&
              (identity - Eigen::Matrix3d::Identity()).norm() < 2e-12,
          "Affine reproduction");
    for (int a = 0; a < 3; ++a)
    {
        const Point dx = Point::Unit(a) * 1e-6;
        const auto high = approximation(sites, solid, x + dx, 1.8),
                   low = approximation(sites, solid, x - dx, 1.8);
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            near((high[i].value - low[i].value) / 2e-6, shape[i].gradient[a], 1e-8, "Shape derivative");
        }
    }
    rejects([&] { approximation({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}, solid, x, 2.); },
            "Coplanar displacement support accepted");
}
