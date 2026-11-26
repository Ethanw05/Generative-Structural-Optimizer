#include "gso/solvers/gpu_device.hpp"

#include "gso/core/error.hpp"

#include <cuda_runtime.h>

#include <format>

namespace gso::solvers {
namespace {

void checkCuda(cudaError_t status, const char* context)
{
    if (status != cudaSuccess) {
        throw EnvironmentError(std::format("{}: {} ({})", context, cudaGetErrorString(status),
                                           cudaGetErrorName(status)));
    }
}

GpuDevice describe(int index)
{
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, index), "Read GPU properties");
    GpuDevice device;
    device.index = index;
    device.name = properties.name;
    device.memoryBytes = properties.totalGlobalMem;
    device.computeCapabilityMajor = properties.major;
    device.computeCapabilityMinor = properties.minor;
    device.multiprocessors = properties.multiProcessorCount;
    return device;
}

}  // namespace

GpuDevice selectGpu()
{
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && count == 0)) {
        throw EnvironmentError("No NVIDIA GPU was found. GSO's solvers need an NVIDIA GPU.");
    }
    checkCuda(status, "Find NVIDIA GPUs");

    GpuDevice best = describe(0);
    for (int index = 1; index < count; ++index) {
        const GpuDevice candidate = describe(index);
        if (candidate.memoryBytes > best.memoryBytes) {
            best = candidate;
        }
    }
    return best;
}

namespace {

// CUDA encodes versions as 1000 * major + 10 * minor.
std::string formatCudaVersion(int version)
{
    return std::format("{}.{}", version / 1000, (version % 1000) / 10);
}

}  // namespace

std::string cudaBuildVersion()
{
    return formatCudaVersion(CUDART_VERSION);
}

std::string cudaDriverVersion()
{
    int version = 0;
    checkCuda(cudaDriverGetVersion(&version), "Read NVIDIA driver version");
    if (version == 0) {
        throw EnvironmentError("No NVIDIA driver is installed. GSO's solvers need an NVIDIA GPU and its driver.");
    }
    return formatCudaVersion(version);
}

}  // namespace gso::solvers
