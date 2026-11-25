#pragma once

#include <cstddef>
#include <limits>

namespace gso::geometry
{
// tr((I - n*n^T) Hess(phi)), in lattice units. Only tangential second
// differences move the interface. A monotone reparameterization of a plane
// therefore has no curvature. Call only with a complete one-cell halo.
inline float tangentialLaplacian(const float* p, std::ptrdiff_t x, std::ptrdiff_t y)
{
    const float gx = .5f * (p[x] - p[-x]);
    const float gy = .5f * (p[y] - p[-y]);
    const float gz = .5f * (p[1] - p[-1]);
    const float norm2 = gx * gx + gy * gy + gz * gz;

    const float xx = p[x] + p[-x] - 2.f * p[0];
    const float yy = p[y] + p[-y] - 2.f * p[0];
    const float zz = p[1] + p[-1] - 2.f * p[0];
    if (norm2 <= std::numeric_limits<float>::min())
    {
        // At a critical point no normal is distinguished. The spherical mean
        // of I - n*n^T is (2/3)I in 3D and is exact for an isotropic Hessian.
        return (2.f / 3.f) * (xx + yy + zz);
    }
    const float xy = .25f * (p[x+y] + p[-x-y] - p[x-y] - p[-x+y]);
    const float xz = .25f * (p[x+1] + p[-x-1] - p[x-1] - p[-x+1]);
    const float yz = .25f * (p[y+1] + p[-y-1] - p[y-1] - p[-y+1]);
    return ((gy*gy + gz*gz)*xx + (gx*gx + gz*gz)*yy + (gx*gx + gy*gy)*zz
            - 2.f * (gx*gy*xy + gx*gz*xz + gy*gz*yz)) / norm2;
}
}
