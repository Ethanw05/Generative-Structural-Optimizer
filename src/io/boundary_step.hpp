#pragma once
#include <filesystem>

namespace gso
{
void writeSurfaceMotion(const std::filesystem::path& input, const std::filesystem::path& output,
                        double moveBound, double availableVolume);
void writeBoundaryStep(const std::filesystem::path& input, const std::filesystem::path& output, double scale);
}
