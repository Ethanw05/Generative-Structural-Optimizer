#pragma once
#include <filesystem>

namespace gso
{
// Reads a triangulated boundary with per-face sensitivities and writes the bounded normal motion per vertex.
void writeSurfaceMotion(const std::filesystem::path& input, const std::filesystem::path& output,
                        double moveBound, double availableVolume);
}
