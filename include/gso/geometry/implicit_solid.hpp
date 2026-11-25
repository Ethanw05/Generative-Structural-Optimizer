#pragma once
#include <Eigen/Core>
#include <vector>

namespace gso
{
using Point = Eigen::Vector3d;
struct Kernel
{
    Point center;
    double radius;
    double amplitude;
};
struct Box
{
    Point low;
    Point high;
    bool contains(const Point &x) const;
};
enum class Occupancy
{
    empty,
    full,
    cut
};
class ImplicitSolid
{
  public:
    ImplicitSolid(Box domain, double background, std::vector<Kernel> kernels = {},
                  std::vector<Box> preserves = {}, std::vector<Box> exclusions = {});
    bool contains(const Point &x) const;
    Occupancy classify(const Box &box) const;
    const Box &bounds() const
    {
        return domain_;
    }
    // Conservative visibility prevents a displacement basis from joining opposite sides of a void.
    bool visible(const Point &a, const Point &b) const;

  private:
    bool segmentDefinitelySolid(const Point &a, const Point &b) const;
    Box domain_;
    double background_;
    std::vector<Kernel> kernels_;
    std::vector<Box> preserves_;
    std::vector<Box> exclusions_;
};
double wendland(double normalizedDistance);
} // namespace gso
