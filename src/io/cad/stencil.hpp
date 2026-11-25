#pragma once
#include <filesystem>
namespace gso {
void writeMlsStencil(double pitch,double modulus,double poisson,const std::filesystem::path& output);
}
