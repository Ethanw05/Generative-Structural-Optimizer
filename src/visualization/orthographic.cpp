#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(_WIN32)
#define GSO_VIEW_EXPORT __declspec(dllexport)
#else
#define GSO_VIEW_EXPORT __attribute__((visibility("default")))
#endif

// Orthographic sample coverage at pixel centers; the greatest view-space Z is visible.
extern "C" GSO_VIEW_EXPORT int gsoRasterize(std::size_t vertices, const double *xyz, std::size_t faces,
    const std::int32_t *indices, const std::uint8_t *colors, int width, int height, std::uint8_t *rgba) noexcept
{
    if (!xyz || !indices || !colors || !rgba || width < 1 || height < 1 ||
        width > 32768 || height > 32768 || !vertices || !faces ||
        vertices > std::numeric_limits<std::size_t>::max()/3 || faces > std::numeric_limits<std::size_t>::max()/3)
    { return 1; }
    try
    {
        const auto pixels = static_cast<std::size_t>(width)*height;
        std::vector<double> depth(pixels, -std::numeric_limits<double>::infinity());
        std::fill_n(rgba, 4*pixels, std::uint8_t{0});
        for (std::size_t i = 0; i < 3*vertices; ++i) { if (!std::isfinite(xyz[i])) { return 2; } }
        for (std::size_t i = 0; i < faces; ++i)
        {
            const double *p[3]{};
            for (unsigned j = 0; j < 3; ++j)
            {
                const auto vertex = indices[3*i+j];
                if (vertex < 0 || static_cast<std::size_t>(vertex) >= vertices) { return 2; }
                p[j] = xyz + 3*static_cast<std::size_t>(vertex);
            }
            const auto *a=p[0], *b=p[1], *c=p[2];
            const double bx=b[0]-a[0], by=b[1]-a[1], cx=c[0]-a[0], cy=c[1]-a[1];
            const double det=bx*cy-by*cx;
            if (det == 0.) { continue; }
            const auto low = [&](int axis, int length) {
                return static_cast<int>(std::clamp(std::ceil(std::min({a[axis],b[axis],c[axis]})-.5),0.,double(length)));
            };
            const auto high = [&](int axis, int length) {
                return static_cast<int>(std::clamp(std::floor(std::max({a[axis],b[axis],c[axis]})-.5),-1.,double(length-1)));
            };
            const int left=low(0,width), right=high(0,width), top=low(1,height), bottom=high(1,height);
            for (int y=top; y<=bottom; ++y)
            {
                for (int x=left; x<=right; ++x)
                {
                    const double dx=x+.5-a[0], dy=y+.5-a[1];
                    const double u=(dx*cy-dy*cx)/det, v=(bx*dy-by*dx)/det;
                    if (u < -1.e-12 || v < -1.e-12 || u+v > 1.+1.e-12) { continue; }
                    const double z=a[2]+u*(b[2]-a[2])+v*(c[2]-a[2]);
                    const auto pixel=static_cast<std::size_t>(y)*width+x;
                    if (z > depth[pixel])
                    {
                        depth[pixel]=z;
                        std::copy_n(colors+3*i,3,rgba+4*pixel);
                        rgba[4*pixel+3]=255;
                    }
                }
            }
        }
        return 0;
    }
    catch (...) { return 3; }
}
