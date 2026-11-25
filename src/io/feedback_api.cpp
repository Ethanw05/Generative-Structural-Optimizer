#include "gso/optimization/feedback_sweep.hpp"
#include <algorithm>
#include <stdexcept>

#if defined(_WIN32)
#define GSO_EXPORT __declspec(dllexport)
#else
#define GSO_EXPORT __attribute__((visibility("default")))
#endif

using Update = int (*)(std::size_t, std::size_t, const unsigned char*, unsigned char*);

extern "C" GSO_EXPORT int gsoFeedbackSweep(std::size_t cases, std::uint64_t seed, std::uint64_t sweep,
                                            const unsigned char* start, Update update, unsigned char* result) noexcept
{
    if (cases == 0 || start == nullptr || update == nullptr || result == nullptr) { return 1; }
    try
    {
        gso::GeometryDigest initial{};
        std::copy_n(start, initial.size(), initial.begin());
        auto step = [update](std::size_t index, std::size_t ordinal, const gso::GeometryDigest& parent) {
            gso::GeometryDigest child{};
            if (update(index, ordinal, parent.data(), child.data()) != 0)
            {
                throw std::runtime_error("Mechanical update interrupted");
            }
            return child;
        };
        const auto last = gso::feedbackSweep(cases, seed, sweep, initial, step);
        std::copy(last.begin(), last.end(), result);
        return 0;
    }
    catch (...) { return 2; }
}
