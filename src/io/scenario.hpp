#pragma once
#include "gso/mechanics/boundary.hpp"
#include "gso/mechanics/stress.hpp"
#include <json.hpp>
namespace gso
{
Point point(const nlohmann::json &value);
SurfacePatch patch(const nlohmann::json &value);
nlohmann::json evaluateCase(const Model &model, const System &system, const nlohmann::json &load,
                            const StressRecovery *recovery);
} // namespace gso
