#pragma once
#include <json.hpp>
namespace gso
{
nlohmann::json analyze(const nlohmann::json &study);
nlohmann::json runEvolution(const nlohmann::json &family);
} // namespace gso
