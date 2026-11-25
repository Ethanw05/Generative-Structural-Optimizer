#pragma once
#include <filesystem>
#include <iosfwd>
#include <string_view>
namespace gso
{
int runStudy(std::string_view command, const std::filesystem::path &path, std::ostream &output);
}
