#pragma once

#include "gso/study/material.hpp"
#include "study/json_fields.hpp"

namespace gso::study::detail {

// Reads one entry of the study's "materials" object and checks that it is physically
// admissible (positive-definite elasticity, positive strengths, consistent fatigue data).
Material readMaterial(const std::string& name, Fields fields);

}  // namespace gso::study::detail
