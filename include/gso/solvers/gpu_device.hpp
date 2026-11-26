#pragma once

// The NVIDIA GPU that runs GSO's solver kernels (docs/port-plan.md, section 1).

#include <cstddef>
#include <string>

namespace gso::solvers {

struct GpuDevice {
    int index = 0;
    std::string name;
    std::size_t memoryBytes = 0;
    int computeCapabilityMajor = 0;
    int computeCapabilityMinor = 0;
    int multiprocessors = 0;
};

// The device the solvers use: the one with the most memory. Throws EnvironmentError if
// there is no usable NVIDIA GPU or its driver is too old for this build.
GpuDevice selectGpu();

// Version of the CUDA toolkit GSO was built with, e.g. "13.5".
std::string cudaBuildVersion();

// Newest CUDA version the installed NVIDIA driver supports, e.g. "13.1". A driver older
// than the build version still runs GSO through CUDA minor-version compatibility, because
// GSO ships kernels compiled for each GPU architecture rather than relying on the driver
// to compile them.
std::string cudaDriverVersion();

}  // namespace gso::solvers
