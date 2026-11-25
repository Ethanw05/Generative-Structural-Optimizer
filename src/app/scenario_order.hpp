#pragma once
#include <iosfwd>
#include <string_view>

namespace gso
{
void writeScenarioOrder(std::string_view count, std::string_view seed, std::string_view sweep, std::ostream& out);
}
